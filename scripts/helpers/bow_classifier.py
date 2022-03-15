import os
import json
import string
import argparse
import numpy as np

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_fscore_support

from vae import data_utils


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Specify estimate or apply")

    estimate_parser = subparsers.add_parser("estimate")
    estimate_parser.set_defaults(estimate=True, apply=False)
    estimate_parser.add_argument(
        "data_dir", type=str,
        help="Directory containing {train,dev,test}.jsonl")
    estimate_parser.add_argument(
        "logdir", type=str,
        help="Where to log results and save the model.")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.set_defaults(estimate=False, apply=True)
    apply_parser.add_argument("logdir", type=str,
                              help="logdir from estimate command.")
    apply_parser.add_argument("data_dir", type=str,
                              help="Data to predict on.")
    apply_parser.add_argument("outdir", type=str,
                              help="Where to save predictions.")

    args = parser.parse_args()
    if [args.estimate, args.apply] == [False, False]:
        parser.print_help()
    return args


def tokenizer(s):
    toks = data_utils.preprocess_sentences([s])[0]
    toks = [t for t in toks if t not in string.punctuation]
    return toks


def estimate(args):
    outfile = os.path.join(args.logdir, "results.log")
    if os.path.exists(outfile):
        raise OSError(f"Existing results found at '{outfile}'. Aborting.")
    os.makedirs(args.logdir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    tmp = data_utils.get_sentences_labels(train_path)
    train_sents, train_labels, sent_ids, label_counts = tmp

    dev_path = os.path.join(args.data_dir, "dev.jsonl")
    tmp = data_utils.get_sentences_labels(dev_path)
    dev_sents, dev_labels, sent_ids, _ = tmp

    test_path = os.path.join(args.data_dir, "test.jsonl")
    tmp = data_utils.get_sentences_labels(test_path)
    test_sents, test_labels, sent_ids, _ = tmp

    vectorizer = CountVectorizer(stop_words=None, ngram_range=(1, 1),
                                 binary=True, preprocessor=None,
                                 tokenizer=tokenizer)
    X = {}
    X["train"] = vectorizer.fit_transform(train_sents)
    X["dev"] = vectorizer.transform(dev_sents)
    X["test"] = vectorizer.transform(test_sents)

    for lab_name in label_counts.keys():
        if lab_name not in ["uncertainty", "polarity"]:
            continue
        y_train = [train_labels[i][lab_name] for i in range(len(train_sents))]
        y_dev = [dev_labels[i][lab_name] for i in range(len(dev_sents))]
        y_test = [test_labels[i][lab_name] for i in range(len(test_sents))]

        # Checking k in the range(2, 30) found that 20 performs best.
        k = 20
        feature_selector = SelectKBest(f_classif, k=k)
        X_train = feature_selector.fit_transform(X["train"], y_train)
        X_dev = feature_selector.transform(X["dev"])
        X_test = feature_selector.transform(X["test"])

        support = feature_selector.get_support()
        names = np.array(vectorizer.get_feature_names())
        chosen_features = names[support]

        model = BernoulliNB(fit_prior=True)
        # model = LogisticRegression(random_state=10, class_weight="balanced")
        model.fit(X_train, y_train)

        with open(outfile, 'a') as outF:
            outF.write(lab_name + '\n')
            outF.write("Features:\n")
            outF.write(str(chosen_features) + '\n')
            outF.write(f"{'':<10} {'precision':<10} {'recall':<10} {'F1':<10}\n")  # noqa
            for (dataname, xs, y) in [("train", X_train, y_train),
                                      ("dev", X_dev, y_dev),
                                      ("test", X_test, y_test)]:
                preds = model.predict(xs)
                p, r, f, _ = precision_recall_fscore_support(
                    y, preds, average="macro")
                outF.write(f"{dataname:<10} {p:<10.4f} {r:<10.4f} {f:<10.4f}\n")  # noqa
            outF.write('\n')

        save_dir = os.path.join(args.logdir, "models", lab_name)
        os.makedirs(save_dir)
        model_file = os.path.join(save_dir, "model.sav")
        joblib.dump(model, model_file)
        vectorizer_file = os.path.join(save_dir, "vectorizer.sav")
        joblib.dump(vectorizer, vectorizer_file)
        selector_file = os.path.join(save_dir, "feature_selector.sav")
        joblib.dump(feature_selector, selector_file)


def apply(args):
    os.makedirs(args.outdir, exist_ok=False)
    for fname in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        fpath = os.path.join(args.data_dir, fname)
        print(f"Predicting on {fpath}")
        if not os.path.exists(fpath):
            print(f"Warning! Expected data file {fpath}, but didn't find it.")
            continue
        tmp = data_utils.get_sentences_labels(fpath)
        sents, labels, sent_ids, label_counts = tmp

        examples = [{"id": sent_ids[i],
                     "sentence": sents[i],
                     **labels[i]}
                    for i in range(len(sents))]
        model_dir = os.path.join(args.logdir, "models")
        for lab_name in os.listdir(model_dir):
            if lab_name in examples[0].keys():
                print(f"Label {lab_name} already exists in {fpath}. Skipping.")
                continue
            lab_model_dir = os.path.join(model_dir, lab_name)
            model_file = os.path.join(lab_model_dir, "model.sav")
            model = joblib.load(model_file)
            vectorizer_file = os.path.join(lab_model_dir, "vectorizer.sav")
            vectorizer = joblib.load(vectorizer_file)
            selector_file = os.path.join(lab_model_dir, "feature_selector.sav")
            feature_selector = joblib.load(selector_file)

            X = vectorizer.transform(sents)
            X = feature_selector.transform(X)
            preds = model.predict(X)
            probs = model.predict_proba(X).max(axis=1)

            for i in range(len(examples)):
                examples[i].update({lab_name: preds[i],
                                    f"{lab_name}_prob": probs[i]})

        outfile = os.path.join(args.outdir, fname)
        with open(outfile, 'w') as outF:
            for example in examples:
                json.dump(example, outF)
                outF.write('\n')


if __name__ == "__main__":
    args = parse_args()
    if args.estimate is True:
        estimate(args)
    elif args.apply is True:
        apply(args)
