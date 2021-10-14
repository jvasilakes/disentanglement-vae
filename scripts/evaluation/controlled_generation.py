import os
import json
import logging
import argparse
from collections import Counter, defaultdict

import torch
import numpy as np
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
    compute_parser.add_argument("--add_padding_token", action="store_true",
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


def get_source_examples_by_length(labs_batch, lens_batch, dataset,
                                  latent_name, id2labs_df):
    labs = labs_batch[latent_name].flatten().numpy().astype(int)
    labs = dataset.label_encoders[latent_name].inverse_transform(labs)
    lengths = lens_batch.flatten().numpy().astype(int)

    samples = []
    for (lab, length) in zip(labs, lengths):
        opposites = id2labs_df[id2labs_df[latent_name] != lab]
        examples = [dataset.get_by_id(uuid) for uuid in opposites.index]
        np.random.shuffle(examples)  # shuffle so we don't overuse examples
        for example in examples:
            if abs(len(example[0].flatten()) - length) <= 3:
                samples.append(example)
                break
    batch = utils.pad_sequence_denoising(samples)
    return batch


def run_generation(vae, dataloader, params, mean_zs, verbose=False):
    """
    mean_zs is a dict {latent_name: {encoded_label: mean_z_value}}
    """
    vae.eval()
    results = []
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, out_Xbatch, Ybatch, lengths, batch_ids = batch
        batch_size = in_Xbatch.size(0)
        in_Xbatch = in_Xbatch.to(vae.device)
        out_Xbatch = out_Xbatch.to(vae.device)
        lengths = lengths.to(vae.device)
        # trg_output = {"decoder_logits": [batch_size, target_length, vocab_size]  # noqa
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "adv_logits": {latent_name-label_name: [batch_size, n_classes]}  # noqa
        #           "token_predictions": [batch_size, target_length]
        trg_output = vae(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        trg_texts = []
        for t in in_Xbatch:
            toks = utils.tensor2text(t, dataloader.dataset.idx2word,
                                     vae.eos_token_idx)
            trg_texts.append(' '.join(toks))

        for latent_name in vae.discriminators.keys():
            # A bit of a hack that depends on labels being binary...
            opposite_ys = torch.abs(Ybatch[latent_name] - 1.0).int().flatten().tolist()  # noqa
            opposite_zs = torch.tensor([mean_zs[latent_name][y]
                                        for y in opposite_ys],
                                       dtype=torch.float32)
            # Transfer the latent
            trg_params = {latent_name: param.z.clone() for (latent_name, param)
                          in trg_output["latent_params"].items()}
            opposite_zs = opposite_zs.reshape(trg_params[latent_name].size())
            trg_params[latent_name] = opposite_zs.to(vae.device)
            z = torch.cat(list(trg_params.values()), dim=1)
            # Decode from it
            max_length = in_Xbatch.size(-1)
            trans_output = vae.sample(z, max_length=max_length)

            # Log the source and transferred text
            trns_texts = []
            for j in range(batch_size):
                trns_t = trans_output["token_predictions"][j, :]
                trns_toks = utils.tensor2text(trns_t,
                                              dataloader.dataset.idx2word,
                                              vae.eos_token_idx)
                trns_text = ' '.join(trns_toks)
                trns_texts.append(trns_text)

            # Re-encode the output to get the discriminators' predictions.
            trans_batch = trans_output["token_predictions"].to(vae.device)
            output_prime = vae(trans_batch, lengths, teacher_forcing_prob=0.0)

            pred_data = [{} for _ in range(batch_size)]
            for (lat_name, dsc) in vae.discriminators.items():
                # If lat_name == latent_name, we want preds == opposite_ys
                # Otherwise we want preds == trg_Ybatch[latent_name]
                preds = dsc.predict(output_prime["dsc_logits"][lat_name])
                preds = preds.detach().cpu().flatten().int().tolist()
                if lat_name == latent_name:
                    true_labs = opposite_ys
                else:
                    true_labs = Ybatch[lat_name].flatten().int().tolist()
                # Log the predictions
                for j in range(batch_size):
                    pred_data[j][lat_name] = {"target": true_labs[j],
                                              "output": preds[j]}
            for j in range(batch_size):
                row = {"transferred_latent": latent_name,
                       "input": trg_texts[j],
                       "output": trns_texts[j],
                       "predictions": pred_data[j]}
                results.append(row)
        if verbose is True:
            pbar.update(1)
        else:
            print(f"{i}/{len(dataloader)}", flush=True)
    if verbose is True:
        pbar.close()

    return results


def add_word_to_sentences(sents, labels):
    ext_sents = []
    word = "unk"
    for (sent, lab) in zip(sents, labels):
        add_word = False
        if lab["polarity"] == "positive":
            add_word = True
        if lab["uncertainty"] == "certain":
            add_word = True
        if add_word is False:
            ext_sents.append(sent)
        else:
            sent.insert(-2, word)  # insert before EOS and presumed punctuation
            ext_sents.append(sent)
    return ext_sents


def compute(args):
    logging.basicConfig(level=logging.INFO)

    SOS = "<SOS>"
    EOS = "<EOS>"

    params = json.load(open(args.params_file, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])
    logdir = os.path.join("logs", params["name"])
    metadata_dir = os.path.join(logdir, "metadata")

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
    if args.add_padding_token is True:
        train_sents = add_word_to_sentences(train_sents, train_labs)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    # Read validation data
    if args.dataset in ["dev", "test"]:
        eval_file = os.path.join(params["data_dir"], f"{args.dataset}.jsonl")
        print(f"Evaluating on {eval_file}")
        tmp = data_utils.get_sentences_labels(eval_file, label_keys=label_keys)
        eval_sents, eval_labs, eval_ids, eval_lab_counts = tmp
        eval_sents = data_utils.preprocess_sentences(eval_sents, SOS, EOS)
        if args.add_padding_token is True:
            eval_sents = add_word_to_sentences(eval_sents, eval_labs)
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

    if args.dataset in ["dev", "test"]:
        data = data_utils.DenoisingTextDataset(
                eval_sents, eval_sents, eval_labs, eval_ids,
                word2idx, label_encoders)
        dataloader = torch.utils.data.DataLoader(
                data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=utils.pad_sequence_denoising)

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

    mean_zs = get_mean_z_per_label(train_labs, train_ids, label_encoders,
                                   metadata_dir, start_epoch - 1)
    results = run_generation(vae, dataloader, params, mean_zs, args.verbose)

    with open(args.outfile, 'w') as outF:
        for row in results:
            json.dump(row, outF)
            outF.write('\n')


def get_mean_z_per_label(train_labs, train_ids, label_encoders,
                         metadata_dir, epoch):
    zdir = os.path.join(metadata_dir, 'z')
    latent_names = set([key for lab in train_labs for key in lab.keys()])
    id_file = os.path.join(metadata_dir, "ordered_ids", f"train_{epoch}.log")
    ordered_ids = [line.strip() for line in open(id_file)]
    id2lab = dict(zip(train_ids, train_labs))
    ordered_labs = [id2lab[uuid] for uuid in ordered_ids]

    mean_zs = defaultdict(dict)
    for latent_name in latent_names:
        lab_vals = set(lab[latent_name] for lab in train_labs)
        zfile = os.path.join(zdir, f"train_{latent_name}_{epoch}.log")
        zs = np.loadtxt(zfile, delimiter=',')
        for lab_val in lab_vals:
            tlabs = np.array([tlab[latent_name] for tlab in ordered_labs])
            idxs = np.argwhere(tlabs == lab_val)
            mean_z = np.mean(zs[idxs])
            lab_val_enc = label_encoders[latent_name].transform([lab_val])[0]
            mean_zs[latent_name][lab_val_enc] = torch.tensor(mean_z)
    return mean_zs


def summarize(args):
    results = [json.loads(line) for line in open(args.outfile)]

    predictions = defaultdict(lambda: defaultdict(list))
    for result in results:
        # The latent that was transferred
        latent = result["latent"]
        for (label_type, preds) in result["predictions"].items():
            true = preds["true"]
            pred = preds["pred"]
            predictions[latent][label_type].append(true == pred)

    print()
    for (trns_latent, label_type_preds) in predictions.items():
        print(f"   Transfering {trns_latent}")
        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|  Prediction  |  Accuracy  |")
        print("|---------------------------|")
        for (label_type, preds) in label_type_preds.items():
            acc = sum(preds) / len(preds)
            print(f"|{label_type:^14}|{acc:^12.4f}|")
        print(" --------------------------- ")
        print()


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "compute":
        compute(args)
    elif args.cmd == "summarize":
        summarize(args)
