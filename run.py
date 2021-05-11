# Built-in packages
import os
import csv
import json
import time
import logging
import argparse
import datetime
from pprint import pformat
from collections import defaultdict

# External packages
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local packages
from vae import utils, data_utils, model, losses


torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(0))
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


class LossLogger(object):

    def __init__(self, summary_writer, epoch):
        self.losses = {}
        self.summary_writer = summary_writer
        self.epoch = epoch

    def __repr__(self):
        return str(self.losses)

    def __str__(self):
        return pformat(self.losses)

    def __getitem__(self, key):
        return self.losses[key]

    def update(self, d, subdict=None):
        """
        Update self.losses with dict d
        """
        to_update = self.losses if subdict is None else subdict
        for (key, val) in d.items():
            if isinstance(val, dict):
                if key not in to_update.keys():
                    to_update[key] = {}
                self.update(val, subdict=to_update[key])
            else:
                if key not in to_update.keys():
                    to_update[key] = []
                val = self._to_scalar(val)
                to_update[key].append(val)

    def _log(self, i, subdict=None, base_keystr='',
             collapse_fn=None, collapse_fn_args=[]):
        if collapse_fn is None:
            raise NotImplementedError("Need to specify a collapse_fn")
        to_log = self.losses if subdict is None else subdict
        for (key, val) in to_log.items():
            keystr = f"{base_keystr}_{key}"
            if isinstance(val, dict):
                self._log(i, subdict=to_log[key], base_keystr=keystr,
                          collapse_fn=collapse_fn,
                          collapse_fn_args=collapse_fn_args)
            elif isinstance(val, list):
                val = self._to_scalar(val)
                logval = collapse_fn(val, *collapse_fn_args)
                self.summary_writer.add_scalar(keystr, logval, i)
            else:
                raise ValueError("Encountered lone scalar '{keystr}: {val}' in LossLogger.log")  # noqa

    def log_epoch(self, subdict=None, base_keystr="avg"):
        self._log(i=self.epoch, subdict=subdict, base_keystr=base_keystr,
                  collapse_fn=np.mean)

    def log_step(self, step, subdict=None, base_keystr="step"):
        self._log(i=step, subdict=subdict, base_keystr=base_keystr,
                  collapse_fn=list.__getitem__, collapse_fn_args=[-1])

    def summarize(self, key):
        val = self.losses[key]
        val = self._to_scalar(val)
        return np.mean(val), np.std(val)

    @classmethod
    def _to_scalar(cls, xs):
        try:
            out = []
            for x in xs:
                out.append(cls._to_scalar(x))
            return out
        except TypeError:
            if isinstance(xs, torch.Tensor):
                return xs.cpu().detach().item()
            elif isinstance(xs, np.ndarray):
                return xs.item()
            else:
                return xs


def safe_dict_update(d1, d2):
    for (key, val) in d2.items():
        if key not in d1.keys():
            d1.update({key: val})


def compute_all_losses(model, model_outputs, Xbatch, Ybatch,
                       lengths, kl_weights_dict, mi_loss_weight):
    # model_outputs = {
    #   "decoder_logits": [batch_size, target_length, vocab_size],
    #   "latent_params": [Params(z, mu, logvar)] * batch_size,
    #   "dsc_logits": {latent_name: [batch_size, n_classes]},
    #   "adv_logits": {adversary_name: [batch_size, n_classes]},
    #   "token_predictions": [batch_size, target_length]}
    L = dict()
    safe_dict_update(
        L, losses.reconstruction_loss(
            Xbatch, model_outputs["decoder_logits"], lengths)
    )

    safe_dict_update(
        L, losses.compute_kl_divergence_losses(
            model, model_outputs["latent_params"], kl_weights_dict)
    )
    safe_dict_update(
        L, losses.compute_discriminator_losses(
            model, model_outputs["dsc_logits"], Ybatch)
    )
    safe_dict_update(
        L, losses.compute_adversarial_losses(
            model, model_outputs["adv_logits"], Ybatch)
    )
    safe_dict_update(
        L, losses.compute_mi_losses(
            model, model_outputs["latent_params"], beta=mi_loss_weight)
    )
    total_loss = (L["reconstruction_loss"] +
                  L["total_weighted_kl"] +
                  L["total_dsc_loss"] +
                  L["total_adv_loss"] +
                  L["total_mi"])
    return total_loss, L


def log_params(params_dict, example_ids, logdir, dataset_name, epoch):
    """
    :param defaultdict params_dict: {latent_name: {parameter: [p1...pN]}}
    :param str logdir:
    :param str dataset_name:
    :param int epoch:
    """
    metadata_dir = os.path.join(logdir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Log example IDs in the same order as their parameters.
    ids_dir = os.path.join(metadata_dir, "ordered_ids")
    os.makedirs(ids_dir, exist_ok=True)
    ids_outfile = os.path.join(ids_dir, f"{dataset_name}_{epoch}.log")
    with open(ids_outfile, 'w') as outF:
        for i in example_ids:
            outF.write(f"{i}\n")

    for latent_name in params_dict.keys():
        for (param_name, values) in params_dict[latent_name].items():
            param_dir = os.path.join(metadata_dir, param_name)
            os.makedirs(param_dir, exist_ok=True)
            outfile = os.path.join(
                    param_dir, f"{dataset_name}_{latent_name}_{epoch}.log")
            with open(outfile, 'w') as outF:
                writer = csv.writer(outF, delimiter=',')
                for value in values:
                    row = [f"{dim:.4f}" for dim in value]
                    writer.writerow(row)


def trainstep(model, optimizer, dataloader, params, epoch, idx2word,
              verbose=True, summary_writer=None, logdir=None):

    epoch_start = time.time()

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    loss_logger = LossLogger(summary_writer, epoch)
    # Log example IDs in same order as latent parameters
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.train()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    step = epoch * len(dataloader)
    total_steps = params["epochs"] * len(dataloader)
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        # output = {"decoder_logits": [batch_size, target_length, vocab_size]
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
        #           "token_predictions": [batch_size, target_length]
        output = model(in_Xbatch, lengths,
                       teacher_forcing_prob=params["teacher_forcing_prob"])

        kl_weights_dict = {}
        for (latent_name, weight) in params["lambdas"].items():
            weight_val = weight
            if weight_val == "cyclic":
                weight_val = losses.get_cyclic_kl_weight(step, total_steps)
            kl_weights_dict[latent_name] = weight_val
            loss_logger.update({"kl_weights": kl_weights_dict})

        # DO NOT CHANGE MI LOSS WEIGHT! IT WORKS NOW BUT WONT IF YOU CHANGE IT!
        mi_loss_weight = 0.01
        loss_logger.update({"mi_loss_weight": mi_loss_weight})

        # COMPUTE MANY MANY LOSSES
        total_loss, losses_dict = compute_all_losses(
            model, output, target_Xbatch, Ybatch,
            lengths, kl_weights_dict, mi_loss_weight)
        loss_logger.update({"total_loss": total_loss})
        loss_logger.update(losses_dict)

        # Update the model
        # I don't exactly know why I need to call backward, update all
        # the adversaries, and then call step(), but it works and I've
        # checked that everything updates properly.
        # with utils.AutogradDebugger():  # uncomment for interactive debugging
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        key = "idv_adv_dsc_losses"
        for (adv_name, adv_dsc_loss) in losses_dict[key].items():
            # Update only the adversaries
            # with utils.AutogradDebugger():
            model.adversaries[adv_name].optimizer_step(adv_dsc_loss)
        optimizer.step()
        optimizer.zero_grad()

        # Update the MI estimators
        key = "idv_mi_estimates"
        for (latent_pair_name, mi_loss) in losses_dict[key].items():
            mi_estimator = model.mi_estimators[latent_pair_name]
            mi_estimator.train()
            latent_name_1, latent_name_2 = latent_pair_name.split('-')
            params1 = output["latent_params"][latent_name_1]
            params2 = output["latent_params"][latent_name_2]
            mi_loss = mi_estimator.learning_loss(
                params1.z.detach(), params2.z.detach())
            mi_estimator.optimizer_step(mi_loss)
            loss_logger.update({"mi_estimator_loss": {latent_pair_name: mi_loss}})  # noqa
            mi_estimator.eval()

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        # Measure Autoencoding by reencoding the reconstructed output.
        x_prime = output["token_predictions"].to(model.device)
        output_prime = model(
                x_prime, lengths,
                teacher_forcing_prob=params["teacher_forcing_prob"])

        for (l_name, l_params) in output_prime["latent_params"].items():
            orig_z = output["latent_params"][l_name].z
            z_prime = l_params.z
            diff = torch.norm(z_prime - orig_z, p=None, dim=1).mean()
            loss_logger.update({"idv_ae": {l_name: diff.item()}})

        # Measure self-BLEU
        bleu = losses.compute_bleu(
            target_Xbatch, x_prime, idx2word, model.eos_token_idx)
        loss_logger.update({"bleu": bleu})

        loss_logger.log_step(step)
        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"EPOCH: {epoch}")
        # After 20 step warmup, get an estimate of time to epoch completion
        if step == (epoch * len(dataloader)) + 20 and verbose is False:
            time_so_far = time.time() - epoch_start
            seconds_2_completion = time_so_far * (len(dataloader) / 20)
            estimated_timedelta = str(datetime.timedelta(
                seconds=seconds_2_completion))
            logstr = f"Estimated epoch duration: {estimated_timedelta}"
            logging.info(logstr)

        step += 1

    if verbose is True:
        pbar.close()

    epoch_time = time.time() - epoch_start
    difftime_str = str(datetime.timedelta(seconds=epoch_time))

    loss_logger.log_epoch()
    log_params(all_latent_params, all_sent_ids, logdir, "train", epoch)

    tlmu, tlsig = loss_logger.summarize("total_loss")
    rcmu, rcsig = loss_logger.summarize("reconstruction_loss")
    klmu, klsig = loss_logger.summarize("total_kl")
    dscmu, dscsig = loss_logger.summarize("total_dsc_loss")
    advmu, advsig = loss_logger.summarize("total_adv_loss")
    mimu, misig = loss_logger.summarize("total_mi")

    logstr = f"TRAIN ({epoch}) TOTAL: {tlmu:.4f} +/- {tlsig:.4f}"
    logstr += f" | RECON: {rcmu:.4f} +/- {rcsig:.4f}"
    logstr += f" | KL: {klmu:.4f} +/- {klsig:.4f}"
    logstr += f" | DISCRIM: {dscmu:.4f} +/- {dscsig:.4f}"
    if model.adversarial_loss is True:
        logstr += f" | ADVERSE: {advmu:.4f} +/- {advsig:.4f}"
    if model.mi_loss is True:
        logstr += f" | MI: {mimu:.4f} +/- {misig:.4f}"
    logstr += f" | Epoch time: {difftime_str}"
    logging.info(logstr)

    return model, optimizer


def evalstep(model, dataloader, params, epoch, idx2word, name="dev",
             verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    loss_logger = LossLogger(summary_writer, epoch)
    # Log example IDs and latent params
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.eval()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        kl_weights_dict = {}
        for (latent_name, weight) in params["lambdas"].items():
            weight_val = weight
            # During evaluation we don't want cyclic annealing.
            if weight_val == "cyclic":
                weight_val = 1.0  # Don't weight it on eval.
            kl_weights_dict[latent_name] = weight_val

        mi_loss_weight = 1.0
        total_loss, losses_dict = compute_all_losses(
            model, output, target_Xbatch, Ybatch, lengths,
            kl_weights_dict, mi_loss_weight)
        loss_logger.update({"total_loss": total_loss})
        loss_logger.update(losses_dict)

        # Measure self-BLEU
        x_prime = output["token_predictions"].to(model.device)
        bleu = losses.compute_bleu(target_Xbatch, x_prime, idx2word,
                                   model.eos_token_idx)
        loss_logger.update({"bleu": bleu})

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f" ↳ EVAL ({name})")

    if verbose is True:
        pbar.close()

    loss_logger.log_epoch()
    log_params(all_latent_params, all_sent_ids, logdir, name, epoch)

    tlmu, tlsig = loss_logger.summarize("total_loss")
    rcmu, rcsig = loss_logger.summarize("reconstruction_loss")
    klmu, klsig = loss_logger.summarize("total_kl")
    dscmu, dscsig = loss_logger.summarize("total_dsc_loss")
    advmu, advsig = loss_logger.summarize("total_adv_loss")
    mimu, misig = loss_logger.summarize("total_mi")

    logstr = f"{name.upper()} ({epoch}) TOTAL: {tlmu:.4f} +/- {tlsig:.4f}"
    logstr += f" | RECON: {rcmu:.4f} +/- {rcsig:.4f}"
    logstr += f" | DISCRIM: {dscmu:.4f} +/- {dscsig:.4f}"
    logstr += f" | KL: {klmu:.4f} +/- {klsig:.4f}"
    if model.adversarial_loss is True:
        logstr += f" | ADVERSE: {advmu:.4f} +/- {advsig:.4f}"
    if model.mi_loss is True:
        logstr += f" | MI: {mimu:.4f} +/- {misig:.4f}"
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
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    # Set model checkpoint directory
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    label_keys = [lk for lk in params["latent_dims"].keys() if lk != "total"]
    if "combined_dataset" in params.keys():
        label_keys.append("source_dataset")  # We need it for batching
    # Read train data
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    tmp = data_utils.get_sentences_labels(
        train_file, N=params["num_train_examples"], label_keys=label_keys)
    train_sents, train_labs, train_ids, train_lab_counts = tmp
    logging.info("Train label counts:")
    for (labname, values) in train_lab_counts.items():
        logging.info(f"  {labname}: {values}")
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    # Read validation data
    dev_file = os.path.join(params["data_dir"], "dev.jsonl")
    tmp = data_utils.get_sentences_labels(dev_file, label_keys=label_keys)
    dev_sents, dev_labs, dev_ids, dev_lab_counts = tmp
    dev_sents = data_utils.preprocess_sentences(dev_sents, SOS, EOS)
    # Use the label encoders fit on the train set
    dev_labs, _ = data_utils.preprocess_labels(
            dev_labs, label_encoders=label_encoders)

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

    if params["reverse_input"] is True:
        noisy_train_sents = data_utils.reverse_sentences(train_sents)
        noisy_dev_sents = data_utils.reverse_sentences(dev_sents)
    else:
        noisy_train_sents = train_sents
        noisy_dev_sents = dev_sents

    # Load glove embeddings, if specified
    # This redefines word2idx/idx2word
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Always load the train data since we need it to build the model
    train_data = data_utils.DenoisingTextDataset(
            noisy_train_sents, train_sents, train_labs, train_ids,
            word2idx, label_encoders)
    dataloader_kwargs = {"shuffle": True, "batch_size": params["batch_size"]}
    if params["combined_dataset"] is True:
        train_sampler = data_utils.RatioSampler(
            train_labs, split_key="source_dataset",
            ratios=params["dataset_minibatch_ratios"],
            batch_size=params["batch_size"])
        dataloader_kwargs = {"batch_sampler": train_sampler}
    train_dataloader = torch.utils.data.DataLoader(
            train_data, collate_fn=utils.pad_sequence_denoising,
            **dataloader_kwargs)

    logging.info(f"Training examples: {len(train_data)}")
    train_writer_path = os.path.join("runs", params["name"], "train")
    train_writer = SummaryWriter(log_dir=train_writer_path)

    if params["validate"] is True:
        dev_data = data_utils.DenoisingTextDataset(
                noisy_dev_sents, dev_sents, dev_labs, dev_ids,
                word2idx, label_encoders)
        dev_dataloader = torch.utils.data.DataLoader(
                dev_data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=utils.pad_sequence_denoising)
        logging.info(f"Validation examples: {len(dev_data)}")
        dev_writer_path = os.path.join("runs", params["name"], "dev")
        dev_writer = SummaryWriter(log_dir=dev_writer_path)

    label_dims_dict = train_data.y_dims
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
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
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")

    # Log the experiment parameter file to recreate this run.
    config_logfile = os.path.join(logdir, f"config_epoch{start_epoch}.json")
    with open(config_logfile, 'w') as outF:
        json.dump(params, outF, indent=2)

    if params["train"] is True:
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and keep most recent model.")
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
                             verbose=verbose, summary_writer=dev_writer,
                             logdir=logdir)
                    # Log dev inputs and their reconstructions
                    utils.log_reconstructions(vae, dev_data, idx2word,
                                              "dev", epoch, logdir, n=20)
                # Save the model
                logging.info(f"Saving model checkpoint to {ckpt_dir}")
                ckpt_fname = f"model_{epoch}.pt"
                ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
                logging.info(f"Saving trained model to {ckpt_path}")
                torch.save({"model_state_dict": vae.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch},
                           ckpt_path)
                checkpoint_found = True
                start_epoch = epoch

            except KeyboardInterrupt:
                logging.warning(f"Training interrupted at epoch {epoch}!")
                break

    if params["validate"] is True:
        evalstep(vae, dev_dataloader, params, start_epoch, idx2word,
                 verbose=verbose, summary_writer=dev_writer, logdir=logdir)
        utils.log_reconstructions(vae, dev_data, idx2word,
                                  "dev", start_epoch, logdir, n=30)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"END: {now_str}")


if __name__ == "__main__":
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
