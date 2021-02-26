# Built-in packages
import os
import re
import csv
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict

# External packages
import torch
import torch.nn.functional as F
import numpy as np
import texar.torch as tx
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score

# Local packages
import utils
import model


torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Path to JSON file containing experiment
                                parameters.""")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="""If specified, print tqdm progress bars
                                for training and evaluation.""")
    return parser.parse_args()


class LabeledTextDataset(torch.utils.data.Dataset):

    def __init__(self, docs, labels, word2idx, label_encoders):
        super(LabeledTextDataset, self).__init__()
        self.docs = docs
        assert isinstance(labels[0], dict)
        self.labels = labels
        if "<UNK>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<UNK>' entry.")
        if "<PAD>" not in word2idx.keys():
            raise ValueError("word2idx must have an '<PAD>' entry.")
        self.word2idx = word2idx
        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
        self.label_encoders = label_encoders
        self.Xs = [self.doc2tensor(doc) for doc in self.docs]
        self.Ys = [self.label2tensor(lab) for lab in self.labels]

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return len(self.Xs)

    @property
    def y_dims(self):
        dims = dict()
        for (label_name, encoder) in self.label_encoders.items():
            num_classes = len(encoder.classes_)
            if num_classes == 2:
                num_classes = 1
            dims[label_name] = num_classes
        return dims

    def doc2tensor(self, doc):
        idxs = []
        for tok in doc:
            try:
                idxs.append(self.word2idx[tok])
            except KeyError:
                idxs.append(self.word2idx["<UNK>"])
        return torch.LongTensor([[idxs]])

    def label2tensor(self, label_dict):
        tensorized = dict()
        for (label_name, label) in label_dict.items():
            encoder = self.label_encoders[label_name]
            # CrossEntropy requires LongTensors
            # BCELoss requires FloatTensors
            if len(encoder.classes_) > 2:
                tensor_fn = torch.LongTensor
            else:
                tensor_fn = torch.FloatTensor
            enc = encoder.transform([label])
            tensorized[label_name] = tensor_fn(enc)
        return tensorized


def get_sentences_labels(path, label_keys=None, N=-1):
    sentences = []
    labels = []
    with open(path, 'r') as inF:
        for (i, line) in enumerate(inF):
            if i == N:
                break
            data = json.loads(line)
            sentences.append(data["sentence"])
            if label_keys is None:
                label_keys = [key for key in data.keys()
                              if key != "sentence"]
            labs = {key: value for (key, value) in data.items()
                    if key in label_keys}
            labels.append(labs)
    return sentences, labels


def preprocess_sentences(sentences, SOS, EOS, lowercase=True):
    out_data = []
    for sent in sentences:
        sent = sent.strip()
        if lowercase is True:
            sent = sent.lower()
        sent = re.sub(r"([.!?])", r" \1", sent)
        sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
        sent = sent.split()
        sent = [SOS] + sent + [EOS]
        out_data.append(sent)
    return out_data


def preprocess_labels(labels, label_encoders={}):
    raw_labels_by_name = defaultdict(list)
    for label_dict in labels:
        for (label_name, lab) in label_dict.items():
            raw_labels_by_name[label_name].append(lab)

    label_encoders = dict()
    enc_labels_by_name = dict()
    for (label_name, labs) in raw_labels_by_name.items():
        if label_name in label_encoders.keys():
            # We're passing in an already fit encoder
            le = label_encoders[label_name]
        else:
            le = LabelEncoder()
        y = le.fit_transform(labs)
        label_encoders[label_name] = le
        enc_labels_by_name[label_name] = y

    return labels, label_encoders


def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    """
    Find the most recent (in epochs) checkpoint in checkpoint dir and load
    it into the model and optimizer. Return the model and optimizer
    along with the epoch the checkpoint was trained to.
    If not checkpoint is found, return the unchanged model and optimizer,
    and 0 for the epoch.
    """
    ls = os.listdir(checkpoint_dir)
    ckpts = [fname for fname in ls if fname.endswith(".pt")]
    if ckpts == []:
        return model, optimizer, 0, None

    latest_ckpt_idx = 0
    latest_epoch = 0
    for (i, ckpt) in enumerate(ckpts):
        epoch = ckpt.replace("model_", '').replace(".pt", '')
        epoch = int(epoch)
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt_idx = i

    ckpt = torch.load(os.path.join(checkpoint_dir, ckpts[latest_ckpt_idx]))
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_epoch = ckpt["epoch"] + 1
    return model, optimizer, next_epoch, ckpts[latest_ckpt_idx]


def tensor2doc(tensor, idx2word, eos_token_idx):
    try:
        eos = torch.where(tensor == eos_token_idx)[0][0]
    except IndexError:
        eos = tensor.size(0)
    return [idx2word[i.item()] for i in tensor[:eos+1]]


def reconstruction_loss(targets, logits, target_lengths):
    recon_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=targets, logits=logits, sequence_length=target_lengths)
    return recon_loss


def kl_divergence(mu, logvar):
    kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    kl = kl.mean(0).sum()
    return kl


def compute_losses(model, model_outputs, Xbatch, Ybatch, lengths, params):
    recon_loss = reconstruction_loss(
            Xbatch, model_outputs["decoder_logits"], lengths)

    # Loss and accuracy for each discriminator
    idv_dsc_losses = dict()
    idv_dsc_accs = dict()
    # total loss over all discriminators
    # used in backward pass
    total_dsc_loss = torch.tensor(0.0).to(model.device)
    for (dsc_name, dsc_logits) in model_outputs["dsc_logits"].items():
        dsc = model.discriminators[dsc_name]
        targets = Ybatch[dsc_name].to(model.device)
        dsc_loss = dsc.compute_loss(dsc_logits, targets)
        dsc_acc = dsc.compute_accuracy(dsc_logits, targets)
        idv_dsc_losses[dsc_name] = dsc_loss.item()
        idv_dsc_accs[dsc_name] = dsc_acc.item()
        total_dsc_loss += dsc_loss

    # KL for each latent space
    idv_kls = dict()
    # total kl over all latent spaces
    # used in backward pass
    # total_kl = torch.tensor(0.0).to(model.device)
    total_kl = 0.0  # scalar for logging
    # tensor scalar for backward pass
    total_weighted_kl = torch.tensor(0.0).to(model.device)
    for (latent_name, latent_params) in model_outputs["latent_params"].items():
        kl = kl_divergence(latent_params.mu, latent_params.logvar)
        idv_kls[latent_name] = kl.item()
        # NB we weight the KL term here.
        # This is so we can easily plug in learnable weights later
        try:
            weight = params["lambdas"][latent_name]
        except KeyError:
            weight = params["lambdas"]["default"]
        total_weighted_kl += weight * kl
        total_kl += kl.item()

    # Compute loss function and do backward pass/update parameters
    loss = recon_loss + total_dsc_loss + total_weighted_kl
    output = {"total_loss": loss,  # Scalar tensor
              "recon_loss": recon_loss.item(),  # scalar
              "total_dsc_loss": total_dsc_loss.item(),  # scalar
              "idv_dsc_losses": idv_dsc_losses,  # dict
              "idv_dsc_accs": idv_dsc_accs,  # dict
              "total_kl": total_kl,  # scalar
              "idv_kls": idv_kls}  # dict
    return output


def compute_bleu(Xbatch, pred_batch, idx2word, eos_token_idx):
    Xtext = [[tensor2doc(X, idx2word, eos_token_idx)[1:-1]]  # RM SOS and EOS
             for X in Xbatch.cpu().detach()]
    pred_text = [tensor2doc(pred, idx2word, eos_token_idx)[1:-1]
                 for pred in pred_batch.cpu().detach()]
    bleu = bleu_score(pred_text, Xtext)
    return bleu


def log_z_metadata(zs_dict, logdir, dataset_name, epoch):
    metadata_dir = os.path.join(logdir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(os.path.join(metadata_dir, "zs"), exist_ok=True)
    for (latent_name, zs) in zs_dict.items():
        outfile = os.path.join(
                metadata_dir, "zs",
                f"zs_{dataset_name}_{latent_name}_{epoch}.log")
        with open(outfile, 'w') as outF:
            writer = csv.writer(outF, delimiter=',')
            for z_row in zs:
                z_row = [f"{z:.4f}" for z in z_row]
                writer.writerow(z_row)


def trainstep(model, optimizer, dataloader, params, epoch, idx2word,
              verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total discriminator losses over discriminators per step
    total_dsc_losses = []
    # losses, accuracies per discriminator per step
    idv_dsc_losses = defaultdict(list)
    idv_dsc_accs = defaultdict(list)
    # Total WEIGHTED KL over latent spaces per step
    total_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)
    idv_ae_mse = defaultdict(list)
    bleus = []
    zs = defaultdict(list)

    model.train()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    step = epoch * len(dataloader)
    for (i, (Xbatch, Ybatch, lengths)) in enumerate(dataloader):
        Xbatch = Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        # output = {"decoder_logits": out_logits,
        #           "latent_params": latent_params,  # Params(z, mu, logvar)
        #           "dsc_logits": dsc_logits}
        output = model(Xbatch, lengths,
                       teacher_forcing_prob=params["teacher_forcing_prob"])

        losses_dict = compute_losses(model, output, Xbatch,
                                     Ybatch, lengths, params)
        total_loss = losses_dict["total_loss"]
        losses.append(total_loss.item())
        recon_losses.append(losses_dict["recon_loss"])
        total_dsc_losses.append(losses_dict["total_dsc_loss"])
        total_kls.append(losses_dict["total_kl"])
        for (dsc_name, dsc_loss) in losses_dict["idv_dsc_losses"].items():
            dsc_acc = losses_dict["idv_dsc_accs"][dsc_name]
            idv_dsc_losses[dsc_name].append(dsc_loss)
            idv_dsc_accs[dsc_name].append(dsc_acc)
        for (latent_name, latent_kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(latent_kl)

        # Update model
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        # Log latents
        for (l_name, l_params) in output["latent_params"].items():
            z_batch = l_params.z.detach().cpu().numpy()
            for z in z_batch:
                zs[l_name].append(z)

        # Measure Autoencoding by reencoding the reconstructed output.
        x_prime = output["decoder_logits"].argmax(-1)
        output_prime = model(
                x_prime, lengths,
                teacher_forcing_prob=params["teacher_forcing_prob"])

        for (l_name, l_params) in output_prime["latent_params"].items():
            orig_z = output["latent_params"][l_name].z
            z_prime = l_params.z
            mse = F.mse_loss(z_prime, orig_z, reduction="mean")
            idv_ae_mse[l_name].append(mse.item())

        # Measure self-BLEU
        bleu = compute_bleu(Xbatch, x_prime, idx2word, model.eos_token_idx)
        bleus.append(bleu)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"EPOCH: {epoch}")
        if step % 5 == 0:
            summary_writer.add_scalar(
                    "total_loss_step", total_loss.item(), step)
            summary_writer.add_scalar(
                    "recon_loss_step", losses_dict["recon_loss"], step)
            for dsc_name in idv_dsc_losses.keys():
                dsc_loss = idv_dsc_losses[dsc_name][-1]
                dsc_acc = idv_dsc_accs[dsc_name][-1]
                summary_writer.add_scalar(
                        f"dsc_loss_step_{dsc_name}", dsc_loss, step)
                summary_writer.add_scalar(
                        f"dsc_acc_step_{dsc_name}", dsc_acc, step)
            for latent_name in idv_kls.keys():
                kl = idv_kls[latent_name][-1]
                summary_writer.add_scalar(
                        f"kl_step_{latent_name}", kl, step)
        step += 1

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar(
            "avg_dsc_loss_all", np.mean(total_dsc_losses), epoch)
    summary_writer.add_scalar("avg_self_bleu", np.mean(bleus), epoch)
    for dsc_name in idv_dsc_losses.keys():
        avg_dsc_loss = np.mean(idv_dsc_losses[dsc_name])
        avg_dsc_acc = np.mean(idv_dsc_accs[dsc_name])
        summary_writer.add_scalar(
                f"avg_dsc_loss_{dsc_name}", avg_dsc_loss, epoch)
        summary_writer.add_scalar(
                f"avg_dsc_acc_{dsc_name}", avg_dsc_acc, epoch)
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)
        avg_ae_mse = np.mean(idv_ae_mse[latent_name])
        summary_writer.add_scalar(
                f"avg_ae_mse_{latent_name}", avg_ae_mse, epoch)

    log_z_metadata(zs, logdir, "train", epoch)

    logstr = f"TRAIN ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logstr += f" | DISCRIM: {np.mean(total_dsc_losses):.4f} +/- {np.std(total_dsc_losses):.4f}"  # noqa
    logstr += f" | KL: {np.mean(total_kls):.4f} +/- {np.std(total_kls):.4f}"  # noqa
    logging.info(logstr)

    return model, optimizer


def evalstep(model, dataloader, params, epoch, idx2word,
             name="dev", verbose=True, summary_writer=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total discriminator losses over discriminators per step
    total_dsc_losses = []
    # losses, accuracies per discriminator per step
    idv_dsc_losses = defaultdict(list)
    idv_dsc_accs = defaultdict(list)
    # Total WEIGHTED KL over latent spaces per step
    total_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)

    model.eval()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, (Xbatch, Ybatch, lengths)) in enumerate(dataloader):
        Xbatch = Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        output = model(Xbatch, lengths, teacher_forcing_prob=0.0)

        losses_dict = compute_losses(model, output, Xbatch,
                                     Ybatch, lengths, params)
        losses.append(losses_dict["total_loss"].item())
        recon_losses.append(losses_dict["recon_loss"])
        total_dsc_losses.append(losses_dict["total_dsc_loss"])
        total_kls.append(losses_dict["total_kl"])
        for (dsc_name, dsc_loss) in losses_dict["idv_dsc_losses"].items():
            dsc_acc = losses_dict["idv_dsc_accs"][dsc_name]
            idv_dsc_losses[dsc_name].append(dsc_loss)
            idv_dsc_accs[dsc_name].append(dsc_acc)
        for (latent_name, kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(kl)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f" ↳ EVAL ({name})")

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar(
            "avg_dsc_loss_all", np.mean(total_dsc_losses), epoch)
    summary_writer.flush()
    for dsc_name in idv_dsc_losses.keys():
        avg_dsc_loss = np.mean(idv_dsc_losses[dsc_name])
        avg_dsc_acc = np.mean(idv_dsc_accs[dsc_name])
        summary_writer.add_scalar(
                f"avg_dsc_loss_{dsc_name}", avg_dsc_loss, epoch)
        summary_writer.add_scalar(
                f"avg_dsc_acc_{dsc_name}", avg_dsc_acc, epoch)
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)

    logstr = f"{name.upper()} ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logstr += f" | DISCRIM: {np.mean(total_dsc_losses):.4f} +/- {np.std(total_dsc_losses):.4f}"  # noqa
    logging.info(logstr)


def run(params_file, verbose=False):
    SOS = "<SOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    params = json.load(open(params_file, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])

    # Set logging directory
    logdir = os.path.join("logs", params["name"])
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "run.log")
    print(f"Logging to {logfile}")
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # Log parameters
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    # Set model checkpoint directory
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Read train data
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    train_sents, train_labs = get_sentences_labels(train_file, N=100)
    train_sents = preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = preprocess_labels(train_labs)
    # Read validation data
    dev_file = os.path.join(params["data_dir"], "dev.jsonl")
    dev_sents, dev_labs = get_sentences_labels(dev_file, N=-1)
    dev_sents = preprocess_sentences(dev_sents, SOS, EOS)
    # Use the label encoders fit on the train set
    dev_labs, _ = preprocess_labels(dev_labs, label_encoders=label_encoders)

    vocab_path = os.path.join(logdir, "vocab.txt")
    if params["train"] is True:
        # Get token vocabulary
        vocab = [PAD, UNK] + \
            list(sorted({word for doc in train_sents for word in doc}))
        # Save the vocabulary for this experiment
        with open(vocab_path, 'w') as outF:
            for word in vocab:
                outF.write(f"{word}\n")
    else:
        vocab = [word.strip() for word in open(vocab_path)]
    # word2idx/idx2word are used for encoding/decoding tokens
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}

    # Load glove embeddings, if specified
    # This redefines word2idx/idx2word
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Always load the train data
    train_data = LabeledTextDataset(train_sents, train_labs,
                                    word2idx, label_encoders)
    train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=params["batch_size"],
            collate_fn=utils.pad_sequence)
    logging.info(f"Training examples: {len(train_data)}")
    train_writer_path = os.path.join("runs", params["name"], "train")
    train_writer = SummaryWriter(log_dir=train_writer_path)

    if params["validate"] is True:
        dev_data = LabeledTextDataset(dev_sents, dev_labs,
                                      word2idx, label_encoders)
        dev_dataloader = torch.utils.data.DataLoader(
                dev_data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=utils.pad_sequence)
        logging.info(f"Validation examples: {len(dev_data)}")
        dev_writer_path = os.path.join("runs", params["name"], "dev")
        dev_writer = SummaryWriter(log_dir=dev_writer_path)

    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    label_dims_dict = train_data.y_dims
    vae = model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    logging.info(vae)

    optimizer = torch.optim.Adam(vae.trainable_parameters(),
                                 lr=params["learn_rate"])

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training/evaluating from there.
    # If no checkpoints exist at checkpoint_dir, load_latest_checkpoint
    # will return the same model and opt, and start_epoch=0
    checkpoint_found = False
    logging.info("Trying to load latest model checkpoint from")
    logging.info(f"  {ckpt_dir}")
    vae, optimizer, start_epoch, ckpt_fname = load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")

    if params["train"] is True:
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")

        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                vae, optimizer = trainstep(
                        vae, optimizer, train_dataloader, params, epoch,
                        idx2word, verbose=verbose, summary_writer=train_writer,
                        logdir=logdir)
                # Log train inputs and their reconstructions
                utils.log_reconstructions(vae, train_data, idx2word,
                                          "train", epoch, logdir, n=20)

                if params["validate"] is True:
                    evalstep(vae, dev_dataloader, params, epoch, idx2word,
                             verbose=verbose, summary_writer=dev_writer)
                    # Log dev inputs and their reconstructions
                    utils.log_reconstructions(vae, dev_data, idx2word,
                                              "dev", epoch, logdir, n=20)

            except KeyboardInterrupt:
                break

        # Save the model
        ckpt_fname = f"model_{epoch}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        logging.info(f"Saving trained model to {ckpt_path}")
        torch.save({"model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                   ckpt_path)
        checkpoint_found = True
        start_epoch = epoch

    if params["validate"] is True:
        evalstep(vae, dev_dataloader, params, start_epoch, idx2word,
                 verbose=verbose, summary_writer=dev_writer)
        utils.log_reconstructions(vae, dev_data, idx2word,
                                  "dev", start_epoch, logdir, n=30)

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"END: {now_str}")


if __name__ == "__main__":
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
