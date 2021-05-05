import os
import argparse
from collections import defaultdict

import nltk
import numpy as np
from tqdm import tqdm

from vae import data_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs='+', type=str,
                        help="""Directory containing
                        {train,dev,test}.jsonl""")
    parser.add_argument("-n", type=int, default=2,
                        help="ngram order. Default 2")
    args = parser.parse_args()
    return args


def estimate(args):
    all_train_sents = {}
    all_dev_sents = {}
    all_test_sents = {}
    print("Loading data.")
    for data_dir in args.data_dirs:
        train_path = os.path.join(data_dir, "train.jsonl")
        dev_path = os.path.join(data_dir, "dev.jsonl")
        test_path = os.path.join(data_dir, "test.jsonl")

        train_sents, _, _, _ = data_utils.get_sentences_labels(train_path)
        train_sents = data_utils.preprocess_sentences(
            train_sents, "<SOS>", "<EOS>")
        all_train_sents[data_dir] = train_sents

        dev_sents, _, _, _ = data_utils.get_sentences_labels(dev_path)
        dev_sents = data_utils.preprocess_sentences(
            dev_sents, "<SOS>", "<EOS>")
        all_dev_sents[data_dir] = dev_sents

        test_sents, _, _, _ = data_utils.get_sentences_labels(test_path)
        test_sents = data_utils.preprocess_sentences(
            test_sents, "<SOS>", "<EOS>")
        all_test_sents[data_dir] = test_sents

    n = args.n
    train_sents = [s for dataset in all_train_sents.values() for s in dataset]
    dev_sents = [s for dataset in all_dev_sents.values() for s in dataset]
    test_sents = [s for dataset in all_test_sents.values() for s in dataset]
    print("Estimating ngram probabilities...")
    train_lm, train_vocab = estimate_ngram_lm(train_sents, n=n)
    print("Perplexities")
    print("train")
    train_ppl, train_num_unks = compute_ppl(train_lm, train_sents, n)
    print("dev")
    dev_ppl, dev_num_unks = compute_ppl(train_lm, dev_sents, n)
    print("test")
    test_ppl, test_num_unks = compute_ppl(train_lm, test_sents, n)
    print(f"TRAIN ngram vocab size: {len(train_vocab)}")
    print(f"TRAIN PPL: {train_ppl:.4f}, UNKS: {train_num_unks}")
    print(f"DEV PPL: {dev_ppl:.4f}, UNKS: {dev_num_unks}")
    print(f"TEST PPL: {test_ppl:.4f}, UNKS: {test_num_unks}")

    if len(all_train_sents) > 1:
        print("\nINDIVIDUAL TRAIN PPLs")
        for (dataset, sents) in all_train_sents.items():
            train_ppl, train_num_unks = compute_ppl(train_lm, sents, n)
            print(f"  {dataset} TRAIN PPL: {train_ppl:.4f}, UNKS: {train_num_unks}")  # noqa
        for (dataset, sents) in all_dev_sents.items():
            dev_ppl, dev_num_unks = compute_ppl(train_lm, sents, n)
            print(f"  {dataset} DEV PPL: {dev_ppl:.4f}, UNKS: {dev_num_unks}")
        for (dataset, sents) in all_test_sents.items():
            test_ppl, test_num_unks = compute_ppl(train_lm, sents, n)
            print(f"  {dataset} TEST PPL: {test_ppl:.4f}, UNKS: {test_num_unks}")  # noqa


def estimate_ngram_lm(sentences, n=1):
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    smoothed_model = defaultdict(lambda: defaultdict(lambda: 1e-8))
    ngram_vocab = set()
    # Count
    for sent in tqdm(sentences):
        ngrams = nltk.ngrams(sent, n)
        for grams in ngrams:
            counts[grams[:-1]][grams[-1]] += 1
            ngram_vocab.add(grams)
    # Normalise
    for indep_gram in counts.keys():
        total_count = sum(counts[indep_gram].values())
        for dep_gram in counts[indep_gram]:
            dep_gram_count = counts[indep_gram][dep_gram]
            smoothed_model[indep_gram][dep_gram] = dep_gram_count / total_count
    return smoothed_model, ngram_vocab


def compute_ppl(model, sentences, n):
    num_unks = 0
    sent_entropies = []
    for sent in tqdm(sentences):
        ngrams = nltk.ngrams(sent, n)
        logprobs = []
        for grams in ngrams:
            prob = model[grams[:-1]][grams[-1]]
            if prob == 1e-8:
                num_unks += 1
            logprobs.append(np.log(prob))
        sent_entropies.append(-np.mean(logprobs))
    ppl = np.exp(np.mean(sent_entropies))
    return ppl, num_unks


def summarize(args):
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    estimate(args)
