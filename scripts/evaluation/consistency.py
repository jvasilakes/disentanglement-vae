import os
import csv
import json
import logging
import argparse
import datetime
from collections import defaultdict

# External packages
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# Local imports
from vae import utils, data_utils, model, losses


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Specify compute, or summarize")

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(compute=True, summarize=False)
    compute_parser.add_argument("params_json", type=str,
                                help="""Path to JSON file containing
                                        experiment parameters.""")
    compute_parser.add_argument("outdir", type=str,
                                help="""Where to save the results.""")
    compute_parser.add_argument("dataset", type=str,
                                choices=["train", "dev", "test"],
                                help="Dataset to summarize.")
    compute_parser.add_argument("--num_resamples", type=int, default=30,
                                required=False,
                                help="""Number of times to resample Z and
                                        decode for a given input example.""")
    compute_parser.add_argument("--verbose", action="store_true", default=False,  # noqa
                                help="""Show a progress bar.""")

    summ_parser = subparsers.add_parser("summarize")
    summ_parser.set_defaults(compute=False, summarize=True)
    summ_parser.add_argument("outdir", type=str,
                             help="Directory containing results of compute.")
    summ_parser.add_argument("dataset", type=str,
                             choices=["train", "dev", "test"],
                             help="Dataset to summarize.")

    return parser.parse_args()


def compute(args):
    logging.basicConfig(level=logging.INFO)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")

    SOS = "<SOS>"
    EOS = "<EOS>"

    # Set up the various input and output files.
    params = json.load(open(args.params_json, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])

    logdir = os.path.join("logs", params["name"])

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No model found at {params['checkpoint_dir']}")

    # Load the train data so we can fully specify the model
    logging.info("Loading train dataset")
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    N = params["num_train_examples"]
    if args.dataset in ["dev", "test"]:
        N = -1
    # Only load the generative factors modeled
    label_keys = [key for key in params["latent_dims"].keys()
                  if key != "total"]
    logging.info(f"Measuring model consistency on the following factors: \n {label_keys}")  # noqa
    tmp = data_utils.get_sentences_labels(
        train_file, N=N, label_keys=label_keys)
    train_sents, train_labs, train_ids, train_lab_counts = tmp
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    vocab_file = os.path.join(logdir, "vocab.txt")
    vocab = [word.strip() for word in open(vocab_file)]
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    train_data = data_utils.DenoisingTextDataset(
        train_sents, train_sents, train_labs, train_ids,
        word2idx, label_encoders)
    dataloader = torch.utils.data.DataLoader(
        train_data, shuffle=False, batch_size=params["batch_size"],
        collate_fn=utils.pad_sequence_denoising)

    # Optionally load dev/test dataset
    if args.dataset in ["dev", "test"]:
        eval_file = os.path.join(params["data_dir"], f"{args.dataset}.jsonl")
        logging.info(f"Evaluating on {eval_file}")
        tmp = data_utils.get_sentences_labels(
            eval_file, N=-1, label_keys=label_keys)
        eval_sents, eval_labs, eval_ids, eval_lab_counts = tmp
        eval_sents = data_utils.preprocess_sentences(eval_sents, SOS, EOS)
        eval_labs, _ = data_utils.preprocess_labels(
            eval_labs, label_encoders=label_encoders)
        eval_data = data_utils.DenoisingTextDataset(
            eval_sents, eval_sents, eval_labs, eval_ids,
            word2idx, label_encoders)
        dataloader = torch.utils.data.DataLoader(
            eval_data, shuffle=False, batch_size=params["batch_size"],
            collate_fn=utils.pad_sequence_denoising)

    # Get word embeddings if specified
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Build the VAE
    label_dims_dict = train_data.y_dims
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    vae = model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(
            vae.trainable_parameters(), lr=params["learn_rate"])

    # Load the latest checkpoint, if there is one.
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    logging.info(f"Loaded checkpoint from '{ckpt_fname}'")

    logging.info(f"Successfully loaded model {params['name']}")
    logging.info(vae)
    vae.train()  # So we sample different latents each time.

    true_labels = defaultdict(list)
    list_fn = lambda: [[] for _ in range(args.num_resamples)]  # noqa
    # predictions given the input
    latent_predictions = defaultdict(list_fn)
    # predictions given the re-encoded input
    latent_predictions_hat = defaultdict(list_fn)
    bleus = list_fn()
    if args.verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
        for (label_name, ys) in Ybatch.items():
            true_labels[label_name].extend(ys.tolist())
        in_Xbatch = in_Xbatch.to(vae.device)
        target_Xbatch = target_Xbatch.to(vae.device)
        lengths = lengths.to(vae.device)
        for resample in range(args.num_resamples):
            # output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
            #           "latent_params": [Params(z, mu, logvar)] * batch_size
            #           "dsc_logits": {latent_name: [batch_size, n_classes]}
            #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
            #           "token_predictions": [batch_size, target_length]
            output = vae(in_Xbatch, lengths, teacher_forcing_prob=0.0)

            # Get the discriminators' predictions for each latent space.
            for (label_name, logits) in output["dsc_logits"].items():
                preds = vae.discriminators[label_name].predict(logits)
                latent_predictions[label_name][resample].extend(
                    preds.cpu().tolist())

            # Get the decoded reconstructions ...
            Xbatch_hat = output["token_predictions"]
            condition = (Xbatch_hat == vae.eos_token_idx) | (Xbatch_hat == 0)
            num_pad = torch.where(condition,                    # if
                                  torch.tensor(1),              # then
                                  torch.tensor(0)).sum(axis=1)  # else
            lengths_hat = Xbatch_hat.size(1) - num_pad
            Xbatch_hat = Xbatch_hat.to(vae.device)
            lengths_hat = lengths_hat.to(vae.device)
            # ... and encode them again ...
            output_hat = vae(Xbatch_hat, lengths_hat, teacher_forcing_prob=0.0)

            # Measure self-BLEU
            bleu = losses.compute_bleu(
                target_Xbatch, Xbatch_hat, idx2word, vae.eos_token_idx)
            bleus[resample].append(bleu)

            # ... and get the discriminators' predictions for the new input.
            for (label_name, logits) in output_hat["dsc_logits"].items():
                preds = vae.discriminators[label_name].predict(logits)
                latent_predictions_hat[label_name][resample].extend(
                    preds.cpu().tolist())

        if args.verbose is True:
            pbar.update(1)
        else:
            logging.info(f"{i}/{len(dataloader)}.")

    results = []
    for label_name in latent_predictions.keys():
        trues = np.array(true_labels[label_name])
        preds = np.array(latent_predictions[label_name])
        preds_hat = np.array(latent_predictions_hat[label_name])
        for resample in range(preds.shape[0]):
            p, r, f, _ = precision_recall_fscore_support(
                trues, preds[resample, :], average="macro")
            row = [resample, label_name, "y", "y_hat", p, r, f]
            results.append(row)

            p, r, f, _ = precision_recall_fscore_support(
                trues, preds_hat[resample, :], average="macro")
            row = [resample, label_name, "y", "y_hat_prime", p, r, f]
            results.append(row)

            p, r, f, _ = precision_recall_fscore_support(
                preds[resample, :], preds_hat[resample, :], average="macro")
            row = [resample, label_name, "y_hat", "y_hat_prime", p, r, f]
            results.append(row)

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(
        args.outdir, f"decoder_predictions_{args.dataset}.csv")
    with open(outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter=',')
        writer.writerow(["batch", "sample_num", "label", "true", "pred",
                         "precision", "recall", "F1"])
        for (batch, row) in enumerate(results):
            writer.writerow([batch] + row)

    bleu_outfile = os.path.join(args.outdir, f"self_bleus_{args.dataset}.csv")
    with open(bleu_outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter=',')
        writer.writerow(["batch", "sample_num", "BLEU"])
        for (resample, sample_bleus) in enumerate(bleus):
            for (batch, b) in enumerate(sample_bleus):
                writer.writerow([batch, resample, b])


def summarize(args):
    infile = os.path.join(
        args.outdir, f"decoder_predictions_{args.dataset}.csv")
    df = pd.read_csv(infile)
    summ_df = df.groupby(["label", "true", "pred"]).agg(
        ["mean", "std"]).drop(["sample_num", "batch"], axis="columns")
    print(summ_df.to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    means = df.groupby(["label", "true", "pred"]).mean().drop(
        ["sample_num", "batch"], axis="columns")
    errs = df.groupby(["label", "true", "pred"]).std().drop(
        ["sample_num", "batch"], axis="columns")
    plots = means.plot.barh(yerr=errs, rot=0, subplots=True,
                            ax=ax, sharey=True, layout=(1, 3), legend=False)

    colors = list(matplotlib.colors.TABLEAU_COLORS.values())[:3]
    for plot in plots[0]:
        for (i, bar) in enumerate(plot.patches):
            c = colors[i % 3]
            bar.set_color(c)
            if i < 3:
                bar.set_hatch('/')
                bar.set_edgecolor('k')

    plt.tight_layout()
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    plot_outfile = os.path.join(
        args.outdir, "plots", f"decoder_predictions_{args.dataset}.pdf")
    fig.savefig(plot_outfile, dpi=300)
    plot_outfile = os.path.join(
        args.outdir, "plots", f"decoder_predictions_{args.dataset}.png")
    fig.savefig(plot_outfile, dpi=300)


if __name__ == "__main__":
    args = parse_args()
    if args.compute is True:
        compute(args)
    elif args.summarize is True:
        summarize(args)
