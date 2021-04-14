import os
import re
import json
import random
import argparse
from hashlib import md5

"""
Split positive/negative sentence pairs into single training examples.
Add binary polarity labels.
Fill in the [MASK] token in each sentence:
    for positive sentences, replace [MASK] with the true token, which is given
    for negative sentences, replace [MASK] with another random token.
"""


SEED = 10
random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True,
                        help="Directory containing {train,dev,test}.jsonl")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory in which to save the output.")
    return parser.parse_args()


def process_file(infile, outfile):
    pairs = [json.loads(line) for line in open(infile, 'r')]
    masked_words = list({pair["masked"] for pair in pairs})

    outlines = []
    seen = set()
    skipped = 0
    for pair in pairs:
        sents = (pair["positive"].lower(), pair["negative"].lower())
        if sents in seen:
            skipped += 1
            continue
        seen.add(sents)
        pos, neg = process_pair(pair, masked_words)
        outlines.extend([pos, neg])
    print(f"Skipped {skipped} duplicate examples.")

    with open(outfile, 'w') as outF:
        for line in outlines:
            json.dump(line, outF)
            outF.write('\n')


def tokenize(string):
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    return string.split()


def process_pair(pair, all_masked_words):
    mask_tok = "[MASK]"
    pos_text = pair["positive"]
    neg_text = pair["negative"]
    pos_masked = pair["masked"]
    pos_text = pos_text.replace(mask_tok, pos_masked)
    neg_text = neg_text.replace(mask_tok, pos_masked)
    predicate = pair["predicateType"]
    pos_sent_id = md5(pos_text.encode()).hexdigest()
    neg_sent_id = md5(neg_text.encode()).hexdigest()
    pos = {"sentence": pos_text,
           "id": pos_sent_id,
           "polarity": 1,
           "predicate": predicate,
           "n_tokens": len(tokenize(pos_text))}
    neg = {"sentence": neg_text,
           "id": neg_sent_id,
           "polarity": 0,
           "predicate": predicate,
           "n_tokens": len(tokenize(neg_text))}
    return pos, neg


def main(indir, outdir):
    os.makedirs(outdir)
    filenames = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    for filename in filenames:
        infile = os.path.join(indir, filename)
        outfile = os.path.join(outdir, filename)
        process_file(infile, outfile)


if __name__ == "__main__":
    args = parse_args()
    main(args.indir, args.outdir)
