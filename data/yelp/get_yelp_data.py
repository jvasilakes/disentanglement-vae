import os
import json
import argparse
from hashlib import md5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        raise OSError(f"Outdir {args.outdir} already exists!")
    for split in ["train", "dev", "test"]:
        for label in [0, 1]:
            fpath = os.path.join(args.indir, f"sentiment.{split}.{label}")
            if not os.path.exists(fpath):
                print(f"Missing input file: {fpath}")
            data = process_file(fpath, label)
            outpath = os.path.join(args.outdir, f"{split}.jsonl")
            with open(outpath, 'a') as outF:
                for datum in data:
                    json.dump(datum, outF)
                    outF.write('\n')


def process_file(fpath, label):
    with open(fpath, 'r') as inF:
        seen_ids = set()
        for line in inF:
            sentence = line.strip()
            length = get_sentence_length(sentence)
            sent_hash = md5(sentence.encode()).hexdigest()
            if sent_hash in seen_ids:
                continue
            example = {"id": sent_hash,
                       "sentence": sentence,
                       "n_tokens": length,
                       "sentiment": label}
            yield example


def get_sentence_length(sentence):
    return len(sentence.split())


if __name__ == "__main__":
    args = parse_args()
    main(args)
