import os
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from collections import Counter

"""
Extract the positive/negative sentence pairs from the ConceptNet data.
Create train/dev/test splits from these pairs with the
  specified proportions.
Reformat the JSON for easier reading later.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help=""".jsonl file from the ConceptNet
                                Negated_LAMA subdirectory.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Directory in which to save the splits.""")
    return parser.parse_args()


def main(infile, outdir, dataset_proportions=[0.7, 0.15, 0.15], random_seed=0):
    """
    :param str infile: Path to file containing ConceptNet data
                       from Negated LAMA.
    :param str outdir: Directory in which to save the train/dev/test data.
    :param List[float] dataset_proportions: [train, dev, test] proportions.
    """
    np.random.seed(random_seed)
    os.makedirs(outdir)
    logfile = os.path.join(outdir, "prepare_polarity_data.log")
    logging.basicConfig(filename=logfile, level=logging.INFO)
    now = datetime.now()
    now_str = now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"RUN: {now_str}")
    logging.info(f"Reading data from {os.path.abspath(infile)}")
    logging.info(f"Saving to {os.path.abspath(outdir)}")
    logging.info(f"Random seed: {random_seed}")

    datasets = ["train", "dev", "test"]
    lengths_pos = {"train": [], "dev": [], "test": []}
    lengths_neg = {"train": [], "dev": [], "test": []}
    preds = {"train": [], "dev": [], "test": []}
    negated_pairs = {"train": [], "dev": [], "test": []}
    seen_sents = set()
    with open(infile) as inF:
        for (i, line) in enumerate(inF):
            data = json.loads(line)
            if "negated" in data.keys():
                dataset = np.random.choice(datasets, p=dataset_proportions)
                pos = data["masked_sentences"][0]
                neg = data["negated"][0]
                if pos in seen_sents or neg in seen_sents:
                    continue
                seen_sents.add(pos)
                seen_sents.add(neg)
                d = {"uuid": data["uuid"],
                     "positive": pos,
                     "negative": neg,
                     "masked": data["obj_label"],
                     "predicateType": data["pred"]}
                negated_pairs[dataset].append(d)
                lengths_pos[dataset].append(len(pos))
                lengths_neg[dataset].append(len(neg))
                preds[dataset].append(data["pred"])

    for dataset in datasets:
        outfile = os.path.join(outdir, f"{dataset}.jsonl")
        with open(outfile, 'w') as outF:
            for pair in negated_pairs[dataset]:
                json.dump(pair, outF)
                outF.write('\n')
        logging.info(f"Negated pairs written to: {outfile}")
        summarize(dataset, lengths_pos, lengths_neg, preds)


def summarize(dataset_name, lengths_pos, lengths_neg, predicates):
    mean_len_pos = f"{np.mean(lengths_pos[dataset_name]):.2f}"
    std_len_pos = f"{np.std(lengths_pos[dataset_name]):.2f}"
    mean_len_neg = f"{np.mean(lengths_neg[dataset_name]):.2f}"
    std_len_neg = f"{np.std(lengths_neg[dataset_name]):.2f}"
    preds_counter = Counter(predicates[dataset_name])
    num_preds = f"{len(preds_counter)}"
    pred_types = "\n\t".join(f"{pred}: {count}" for (pred, count)
                             in preds_counter.most_common())
    stats = f"\nNumber of sentence pairs: {len(lengths_pos[dataset_name])}"
    stats += f"\nLengths of + sentences: {mean_len_pos} +/- {std_len_pos}"
    stats += f"\nLengths of - sentences: {mean_len_neg} +/- {std_len_neg}"
    stats += f"\nNum unqiue predicates: {num_preds}"
    stats += "\nPredicate counts:"
    stats += f"\n\t{pred_types}"

    logging.info(f"STATISTICS FOR DATASET: {dataset_name}")
    logging.info(stats)


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outdir)
