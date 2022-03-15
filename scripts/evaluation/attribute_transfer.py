import os
import json
import logging
import argparse
from collections import Counter, defaultdict

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from vae import utils, data_utils, model


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(cmd="compute")
    compute_parser.add_argument("params_file", type=str,
                                help="Parameter file specifying model.")
    compute_parser.add_argument("outfile", type=str,
                                help="Where to save results.")
    compute_parser.add_argument("dataset", type=str,
                                choices=["train", "dev", "test"],
                                help="Which dataset to run on.")
    compute_parser.add_argument("--verbose", action="store_true",
                                default=False)

    summ_parser = subparsers.add_parser("summarize")
    summ_parser.set_defaults(cmd="summarize")
    summ_parser.add_argument("outfile", type=str,
                             help="""outfile from compute command.""")

    return parser.parse_args()


def get_source_examples(labs_batch, dataset, latent_name, id2labs_df):
    labs = labs_batch[latent_name].flatten().numpy().astype(int)
    labs_decoded = dataset.label_encoders[latent_name].inverse_transform(labs)
    encoded_value_counts = Counter(labs_decoded)

    idx2example = {}
    for (value, count) in encoded_value_counts.items():
        encoded_value = dataset.label_encoders[latent_name].transform([value])[0]  # noqa
        # Get the idxs of the examples in the batch that we need to
        #   find source examples for.
        idxs = np.argwhere(labs == encoded_value).flatten()
        # Get the IDs of the corresponding number of source examples
        #  with a different value than the target.
        samples = id2labs_df[id2labs_df[latent_name] != value].sample(count)
        # Get the processed examples corresponding to those IDs.
        examples = [dataset.get_by_id(uuid) for uuid in samples.index]
        for (idx, ex) in zip(idxs, examples):
            idx2example[idx] = ex

    # When the above for loop is finished, we should have an example for each
    # index.
    ordered_examples = [idx2example[i] for i in range(len(idx2example))]
    # So we just have to turn it into a batch that can be fed into the model.
    batch = utils.pad_sequence_denoising(ordered_examples)
    return batch


def run_transfer(model, dataloader, params, id2labs_df, verbose=False):
    model.eval()
    results = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, out_Xbatch, Ybatch, lengths, batch_ids = batch
        batch_size = in_Xbatch.size(0)
        in_Xbatch = in_Xbatch.to(model.device)
        out_Xbatch = out_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        # trg_output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
        #           "token_predictions": [batch_size, target_length]
        trg_output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        trg_texts = []
        for t in in_Xbatch:
            toks = utils.tensor2text(t, dataloader.dataset.idx2word,
                                     model.eos_token_idx)
            trg_texts.append(' '.join(toks))

        for latent_name in model.discriminators.keys():
            src_batch = get_source_examples(
                Ybatch, dataloader.dataset, latent_name, id2labs_df)
            src_Xbatch, _, src_Ybatch, src_lengths, src_batch_ids = src_batch
            src_Xbatch = src_Xbatch.to(model.device)
            src_lengths = src_lengths.to(model.device)
            src_output = model(src_Xbatch, src_lengths, teacher_forcing_prob=0.0)  # noqa

            # Transfer the latent
            trg_params = {latent_name: param.z.clone() for (latent_name, param)
                          in trg_output["latent_params"].items()}
            trg_params[latent_name] = src_output["latent_params"][latent_name].z.clone()  # noqa
            z = torch.cat(list(trg_params.values()), dim=1)
            # Decode from it
            max_length = in_Xbatch.size(-1)
            trans_output = model.sample(z, max_length=max_length)

            # Log the source and transferred text
            src_texts = []
            trns_texts = []
            for j in range(batch_size):
                src_t = src_Xbatch[j, :]
                src_toks = utils.tensor2text(
                    src_t, dataloader.dataset.idx2word, model.eos_token_idx)
                src_text = ' '.join(src_toks)
                src_texts.append(src_text)

                trns_t = trans_output["token_predictions"][j, :]
                trns_toks = utils.tensor2text(trns_t,
                                              dataloader.dataset.idx2word,
                                              model.eos_token_idx)
                trns_text = ' '.join(trns_toks)
                trns_texts.append(trns_text)

            # Re-encode the output to get the discriminators' predictions.
            trans_batch = trans_output["token_predictions"].to(model.device)
            output_prime = model(trans_batch, lengths,
                                 teacher_forcing_prob=0.0)

            pred_data = [{} for _ in range(batch_size)]
            for (lat_name, dsc) in model.discriminators.items():
                # If lat_name == latent_name, we want preds == src_Ybatch[latent_name]  # noqa
                # Otherwise we want preds == trg_Ybatch[latent_name]
                preds = dsc.predict(output_prime["dsc_logits"][lat_name])
                preds = preds.detach().cpu().flatten().int().tolist()
                if lat_name == latent_name:
                    true_labs = src_Ybatch[lat_name].flatten().int().tolist()
                else:
                    true_labs = Ybatch[lat_name].flatten().int().tolist()
                # Log the predictions
                for j in range(batch_size):
                    pred_data[j][lat_name] = {"true": true_labs[j],
                                              "pred": preds[j]}
            for j in range(batch_size):
                row = {"latent": latent_name,
                       "target": trg_texts[j],
                       "source": src_texts[j],
                       "transferred": trns_texts[j],
                       "predictions": pred_data[j]}
                results.append(row)
        if verbose is True:
            pbar.update(1)
        else:
            print(f"{i}/{len(dataloader)}", flush=True)
    if verbose is True:
        pbar.close()

    return results


def compute(args):
    logging.basicConfig(level=logging.INFO)

    SOS = "<SOS>"
    EOS = "<EOS>"

    params = json.load(open(args.params_file, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])
    logdir = os.path.join("logs", params["name"])

    # Set logging directory
    # Set model checkpoint directory
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Read train data
    label_keys = [lk for lk in params["latent_dims"].keys() if lk != "total"]
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    tmp = data_utils.get_sentences_labels(
        train_file, N=params["num_train_examples"], label_keys=label_keys)
    train_sents, train_labs, train_ids, train_lab_counts = tmp
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    # Read validation data
    if args.dataset in ["dev", "test"]:
        eval_file = os.path.join(params["data_dir"], f"{args.dataset}.jsonl")
        print(f"Evaluating on {eval_file}")
        tmp = data_utils.get_sentences_labels(eval_file, label_keys=label_keys)
        eval_sents, eval_labs, eval_ids, eval_lab_counts = tmp
        eval_sents = data_utils.preprocess_sentences(eval_sents, SOS, EOS)
        # Use the label encoders fit on the train set
        eval_labs, _ = data_utils.preprocess_labels(
                eval_labs, label_encoders=label_encoders)

    vocab_path = os.path.join(logdir, "vocab.txt")
    vocab = [word.strip() for word in open(vocab_path)]
    # word2idx/idx2word are used for encoding/decoding tokens
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}

    # Load glove embeddings, if specified
    # This redefines word2idx/idx2word
    emb_matrix = None
    if params["glove_path"] != "":
        print(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        print(f"Loaded embeddings with size {emb_matrix.shape}")
    # idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Always load the train data since we need it to build the model
    data = data_utils.DenoisingTextDataset(
            train_sents, train_sents, train_labs, train_ids,
            word2idx, label_encoders)
    dataloader = torch.utils.data.DataLoader(
            data, collate_fn=utils.pad_sequence_denoising,
            shuffle=False, batch_size=params["batch_size"])
    label_dims_dict = data.y_dims

    if args.dataset == "train":
        labs_df = pd.DataFrame(train_labs, index=train_ids)
    elif args.dataset in ["dev", "test"]:
        data = data_utils.DenoisingTextDataset(
                eval_sents, eval_sents, eval_labs, eval_ids,
                word2idx, label_encoders)
        dataloader = torch.utils.data.DataLoader(
                data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=utils.pad_sequence_denoising)
        labs_df = pd.DataFrame(eval_labs, index=eval_ids)

    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    vae = model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(
        vae.trainable_parameters(), lr=params["learn_rate"])

    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")
    vae, _, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
        vae, optimizer, ckpt_dir, map_location=DEVICE)
    print(f"Loaded checkpoint from '{ckpt_fname}'")
    print(vae)

    results = run_transfer(vae, dataloader, params, labs_df, args.verbose)

    with open(args.outfile, 'w') as outF:
        for row in results:
            json.dump(row, outF)
            outF.write('\n')


def summarize(args):
    results = [json.loads(line) for line in open(args.outfile)]

    predictions = defaultdict(lambda: defaultdict(list))
    for result in results:
        latent = result["latent"]
        for (label_type, preds) in result["predictions"].items():
            true = preds["true"]
            pred = preds["pred"]
            if label_type == latent:
                label_type = f"{label_type}: {str(true)}->{str(abs(1-true))}"
            else:
                label_type = f"{label_type}: {str(true)}"
            predictions[latent][label_type].append(true == pred)

    print()
    for (trns_latent, label_type_preds) in predictions.items():
        print(f"   Transfering {trns_latent}")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|    Prediction      |  Accuracy  |")
        print("|---------------------------------|")
        for (label_type, preds) in label_type_preds.items():
            acc = sum(preds) / len(preds)
            print(f"|{label_type:^20}|{acc:^12.4f}|")
        print(" --------------------------------- ")
        print()


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "compute":
        compute(args)
    elif args.cmd == "summarize":
        summarize(args)
