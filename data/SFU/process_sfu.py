import os
import json
import string
import argparse
import xml.etree.ElementTree as ET
from hashlib import md5
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_dirs", type=str, nargs='+',
                        help="""Directories containing XML files.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the processed files.""")
    parser.add_argument("--max_length", type=int, default=None,
                        help="""Maximum sentence length to keep.""")
    return parser.parse_args()


def main(args):
    out_data = []
    seen_sents = set()
    for corpus_dir in args.corpus_dirs:
        files = os.listdir(corpus_dir)
        review_type = os.path.basename(corpus_dir.rstrip('/'))
        for file in files:
            filepath = os.path.join(corpus_dir, file)
            sentences = read_xml(filepath, attrs={"review_type": review_type})
            for sentence in sentences:
                if args.max_length is not None:
                    if len(sentence["sentence"].split()) > args.max_length:
                        continue
                sent_lower = sentence["sentence"].lower()
                sent_hash = md5(sent_lower.encode()).hexdigest()
                if sent_hash not in seen_sents:
                    sentence["id"] = sent_hash
                    out_data.append(sentence)
                    seen_sents.add(sent_hash)

    np.random.shuffle(out_data)
    train, evals = train_test_split(out_data, test_size=0.3)
    dev, test = train_test_split(evals, test_size=0.5)

    train_file = os.path.join(args.outdir, "train.jsonl")
    dev_file = os.path.join(args.outdir, "dev.jsonl")
    test_file = os.path.join(args.outdir, "test.jsonl")

    with open(train_file, 'w') as outF:
        for datum in train:
            json.dump(datum, outF)
            outF.write('\n')

    with open(dev_file, 'w') as outF:
        for datum in dev:
            json.dump(datum, outF)
            outF.write('\n')

    with open(test_file, 'w') as outF:
        for datum in test:
            json.dump(datum, outF)
            outF.write('\n')

        log_dataset_stats(train, dev, test, names=["train", "dev", "test"])


def read_xml(file, attrs={}):
    try:
        tree = ET.parse(file)
    except ET.ParseError:
        print(file)
        print("not well formed. continuing...")
        return []
    root = tree.getroot()
    out_sentences = []
    for sentence in root.findall(".//SENTENCE"):
        elements = sentence.findall("./*")
        (subwords, sent_attrs) = split_sentence(elements)

        for (words, sent_attr) in zip(subwords, sent_attrs):
            if len(words) <= 2:
                continue
            sent_toks = [w.text for w in words]
            if sent_toks[0] in string.punctuation:
                sent_toks = sent_toks[1:]
            sent_txt = ' '.join(sent_toks)
            sent_txt = filter(sent_txt)
            if sent_txt.strip() == '':
                continue
            # If this sentence was the result of splitting a longer one.
            was_split = False
            if len(subwords) > 1:
                was_split = True
            datum = {"sentence": sent_txt,
                     "was_split": was_split,
                     **sent_attr, **attrs}

            out_sentences.append(datum)
    return out_sentences


def split_sentence(elements):
    """
    Split a single sentence into multiple on any
    <C>and</C> tag.
    """
    sentences = []
    sent_attrs = []
    curr = []
    attrs = {"uncertainty": "certain", "polarity": "positive"}
    for elem in elements:
        if elem.tag == 'W':
            curr.append(elem)
        # Sometimes cues are wrapped in a <C> tag, sometimes not.
        elif elem.tag in ['C', "cue"]:
            if elem.tag == 'C':
                cue = elem.find("./cue")
            else:
                cue = elem

            if cue is not None:
                attr = cue.get("type")
                if attr == "speculation":
                    attrs["uncertainty"] = "uncertain"
                elif attr == "negation":
                    attrs["polarity"] = "negative"
                curr.append(elem.find(".//W"))
            else:
                w = elem.find(".//W")
                if w.text.lower() == "and":
                    sentences.append(curr)
                    curr = []
                    sent_attrs.append(attrs)
                    attrs = {"uncertainty": "certain", "polarity": "positive"}
                    continue
                elif w is not None:
                    curr.append(w)
        else:
            subwords, subattrs = split_sentence(elem.findall("./"))
            if subwords[0]:
                curr.extend(subwords[0])
                if subattrs[0]["uncertainty"] == "uncertain":
                    attrs.update({"uncertainty": "uncertain"})
                if subattrs[0]["polarity"] == "negative":
                    attrs.update({"polarity": "negative"})
    sentences.append(curr)
    sent_attrs.append(attrs)
    return sentences, sent_attrs


def log_dataset_stats(*arrays, names=None):
    if names is None:
        names = range(len(arrays))
    for (arr, name) in zip(arrays, names):
        counts = defaultdict(lambda: defaultdict(int))
        sents = set()
        for ex in arr:
            for key in ["review_type", "uncertainty", "polarity", "was_split"]:
                counts[key][ex[key]] += 1
                sents.add(ex["sentence"])
        print(f"===== {name} =====")
        for (key, sub_dict) in counts.items():
            print(key)
            for (subkey, val) in sorted(sub_dict.items()):
                print(f"  {subkey}: {val}")
        print(f"Unique sents/total: {len(sents)} / {len(arr)}")
        print()


def filter(string):
    return string.encode("ascii", "ignore").decode("utf8")


if __name__ == "__main__":
    args = parse_args()
    main(args)
