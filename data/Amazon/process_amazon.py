import os
import json
import argparse
from hashlib import md5

import numpy as np
from tqdm import tqdm


np.random.seed(10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True,
                        help="Directory containing {pos,neg}.txt")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the processed files.")
    parser.add_argument("--max_length", type=int, default=15,
                        help="Maximum token length to keep.")
    return parser.parse_args()


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    pos_file = os.path.join(args.indir, "pos.txt")
    neg_file = os.path.join(args.indir, "neg.txt")

    pos_sents = open(pos_file, 'r').readlines()
    neg_sents = open(neg_file, 'r').readlines()

    train, dev, test = split_and_process(pos_sents, neg_sents,
                                         max_length=args.max_length)

    train_file = os.path.join(args.outdir, "train.jsonl")
    with open(train_file, 'w') as outF:
        for example in train:
            json.dump(example, outF)
            outF.write('\n')
    dev_file = os.path.join(args.outdir, "dev.jsonl")
    with open(dev_file, 'w') as outF:
        for example in dev:
            json.dump(example, outF)
            outF.write('\n')
    test_file = os.path.join(args.outdir, "test.jsonl")
    with open(test_file, 'w') as outF:
        for example in test:
            json.dump(example, outF)
            outF.write('\n')


def split_and_process(pos_sents, neg_sents, max_length=15):
    # [train, dev, test]
    splits = [[], [], []]
    ps = [0.7, 0.15, 0.15]  # train, dev, test proportions
    seen_ids = set()
    num_duplicates = 0
    num_too_long = 0
    print("Processing positive examples.")
    for pos_sent in tqdm(pos_sents):
        processed = process_sent(pos_sent, labels={"sentiment": 1})
        if processed["id"] in seen_ids:
            num_duplicates += 1
            continue
        if processed["n_tokens"] > max_length:
            num_too_long += 1
            continue
        seen_ids.add(processed["id"])
        split = np.random.choice(range(3), p=ps)
        splits[split].append(processed)
    print("Processing negative examples.")
    for neg_sent in tqdm(neg_sents):
        processed = process_sent(neg_sent, labels={"sentiment": 0})
        if processed["id"] in seen_ids:
            num_duplicates += 1
            continue
        if processed["n_tokens"] > max_length:
            num_too_long += 1
            continue
        seen_ids.add(processed["id"])
        split = np.random.choice(range(3), p=ps)
        splits[split].append(processed)

    print(f"Skipped {num_duplicates} duplicate sentences")
    print(f"Skipped {num_too_long} sentences > {max_length} tokens")
    return splits


def process_sent(sent, labels={}):
    sent = sent.strip()
    length = len(sent.split())
    sent_hash = md5(sent.encode()).hexdigest()
    example = {"id": sent_hash,
               "sentence": sent,
               "n_tokens": length}
    example.update(labels)
    return example


if __name__ == "__main__":
    args = parse_args()
    main(args)
