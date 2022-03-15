import os
import json
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import vae.model as vae_model
from vae import utils, data_utils


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Config file for model to
                                use for reconstruction.""")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing {train,dev,test}.jsonl")
    parser.add_argument("logfile", type=str,
                        help="Where to save the reconstructions.")
    parser.add_argument("-N", type=int, default=-1,
                        help="Number of examples to use.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Use tqdm progress bars.")
    return parser.parse_args()


def main(data_dir, params_json, logfile, max_n=-1, verbose=False):
    logging.info("Loading data...")
    # data = {dataset_name: [sentences]}
    data = get_data(data_dir)

    logging.info("Running reconstruction...")
    recon_data = reconstruct_with_model(data, params_json,
                                        N=max_n, verbose=verbose)

    logging.info("Loading GPT2...")
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(DEVICE)

    logging.info("Computing PPLs...")
    for (dataset_name, sents) in data.items():
        ppl = compute_ppl(sents[:max_n], tokenizer, model, verbose=verbose)
        recon_ppl = compute_ppl(recon_data[dataset_name], tokenizer, model,
                                verbose=verbose)
        logging.info(f"{dataset_name}: {ppl:.4f}")
        logging.info(f"    recon: {recon_ppl:4f}")

    with open(logfile, 'w') as outF:
        for (dataset_name, sents) in data.items():
            for (sent, recon) in zip(sents[:max_n], recon_data[dataset_name]):
                json.dump({"dataset": dataset_name, "sentence": sent,
                           "reconstruction": recon}, outF)
                outF.write('\n')
    logging.info(f"Reconstructions saved to {logfile}")


def compute_ppl(sentences, tokenizer, model, stride=512, verbose=False):
    encodings = tokenizer.encode("\n\n".join(sentences), return_tensors='pt')
    max_length = model.config.n_positions

    nlls = []
    if verbose is True:
        pbar = tqdm(total=len(range(0, encodings.size(1), stride)))
    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i
        input_ids = encodings[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        #target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs[0] * trg_len
        nlls.append(nll)
        if verbose is True:
            pbar.update(1)
        else:
            if i % 10 == 0:
                logging.info(f"{i}/{len(range(0, encodings.size(1), stride))}")
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl


def get_data(data_dir):
    dataset_names = ["train", "dev", "test"]
    output = {}
    for name in dataset_names:
        fname = os.path.join(data_dir, f"{name}.jsonl")
        data = [json.loads(line) for line in open(fname)]
        sentences = [datum["sentence"] for datum in data]
        output[name] = sentences
    return output


def reconstruct_with_model(data, params_json, N=-1, num_resamples=1,
                           verbose=False):
    SOS = "<SOS>"
    EOS = "<EOS>"

    # Set up the various input and output files.
    params = json.load(open(params_json, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])

    logdir = os.path.join("logs", params["name"])

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No model found at {params['checkpoint_dir']}")

    # Load the train data so we can fully specify the model
    dataloaders = {}
    for dataset in ["train", "dev", "test"]:
        data_file = os.path.join(params["data_dir"], f"{dataset}.jsonl")
        tmp = data_utils.get_sentences_labels(data_file, N=N)
        data_sents, data_labs, data_ids, data_lab_counts = tmp
        data_sents = data_utils.preprocess_sentences(data_sents, SOS, EOS)
        data_labs, label_encoders = data_utils.preprocess_labels(data_labs)

        # keep vocab the same for dev/test
        if dataset == "train":
            vocab_file = os.path.join(logdir, "vocab.txt")
            vocab = [word.strip() for word in open(vocab_file)]
            word2idx = {word: idx for (idx, word) in enumerate(vocab)}
        data = data_utils.DenoisingTextDataset(
            data_sents, data_sents, data_labs, data_ids,
            word2idx, label_encoders)
        dataloader = torch.utils.data.DataLoader(
            data, shuffle=False, batch_size=params["batch_size"],
            collate_fn=utils.pad_sequence_denoising)
        dataloaders[dataset] = dataloader

    # Get word embeddings if specified
    emb_matrix = None
    if params["glove_path"] != "":
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Build the VAE
    label_dims_dict = dataloaders["train"].dataset.y_dims
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    vae = vae_model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                              DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(
            vae.trainable_parameters(), lr=params["learn_rate"])

    # Load the latest checkpoint, if there is one.
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir, map_location=torch.device(DEVICE))
    logging.info(f"Loaded checkpoint from '{ckpt_fname}'")
    logging.info(f"Successfully loaded model {params['name']}")
    logging.info(vae)
    vae.eval()

    reconstructed_sentences = defaultdict(list)
    for (name, dataloader) in dataloaders.items():
        if verbose is True:
            pbar = tqdm(total=len(dataloader))
            pbar.set_description(name)
        for (i, batch) in enumerate(dataloader):
            in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
            in_Xbatch = in_Xbatch.to(vae.device)
            target_Xbatch = target_Xbatch.to(vae.device)
            lengths = lengths.to(vae.device)
            for resample in range(num_resamples):
                # output = {
                #   "decoder_logits": [batch_size, target_length, vocab_size]
                #   "latent_params": [Params(z, mu, logvar)] * batch_size
                #   "dsc_logits": {latent_name: [batch_size, n_classes]}
                #   "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
                #   "token_predictions": [batch_size, target_length]
                output = vae(in_Xbatch, lengths, teacher_forcing_prob=0.0)

                # Get the decoded reconstructions ...
                Xbatch_hat = output["token_predictions"]
                texts = [utils.tensor2text(X, idx2word, eos_idx)[1:-1]  # rm SOS and EOS  # noqa
                         for X in Xbatch_hat.cpu().detach()]
                recons = [' '.join(t) for t in texts]
                reconstructed_sentences[name].extend(recons)
            if verbose is True:
                pbar.update(1)
            else:
                if i % 10 == 0:
                    logging.info(f"({name}) {i}/{len(dataloader)}")
    return reconstructed_sentences


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.params_json, args.logfile,
         max_n=args.N, verbose=args.verbose)
