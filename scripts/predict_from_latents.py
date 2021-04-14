import os
import re
import json
import argparse
from glob import glob
from collections import Counter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=str,
                        help="Directory containing z/ and ordered_ids/")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing {train,dev,test.jsonl}.")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Which training epoch to run on. Default last.")
    return parser.parse_args()


def main(args):
    """
    Get zs from metadata_dir/z/train_*_epoch.log
    Get labels from data_dir/train.jsonl using ids from
        metadata_dir/ordered_ids/train_epoch.log
    For each zs/labels pair:
        Estimate logistic regression model from zs to labels.
        Evaluate above model on data_dir/{train,dev,test}.jsonl
    """
    zs_dir = os.path.join(args.metadata_dir, 'z')
    if args.epoch == -1:
        epoch = get_last_epoch(zs_dir)
    else:
        epoch = args.epoch
    z_files = glob(os.path.join(zs_dir, f"train_*_{epoch}.log"))
    latent_names = get_latent_names(z_files)

    ids_dir = os.path.join(args.metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"train_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    # id2labels = {uuid: {latent_name: value for latent_name in latent_names}}
    # But this will not include the "content" latent.
    # labels_set = {lname for lname in latent_names if lname is a supervised latent}  # noqa
    id2labels, labels_set = get_labels(args.data_dir, latent_names)

    for (lname, zfile) in zip(latent_names, z_files):
        for lab_name in labels_set:
            # Predict lab_name from z_lname
            zs = np.loadtxt(zfile, delimiter=',')
            id2z = dict(zip(ids, zs))
            print(f"{lname} |-> {lab_name}")
            results = train_lr(lname, id2z, lab_name, id2labels)
            print(results)
            print()


def get_last_epoch(directory):
    files = os.listdir(directory)
    epochs = {int(re.findall(r'.*_([0-9]+)\.log', fname)[0])
              for fname in files}
    return max(epochs)


def get_latent_names(filenames):
    latent_names = []
    for fname in filenames:
        name = re.findall(r'.*_(\w+)_[0-9]+.log', fname)[0]
        latent_names.append(name)
    return latent_names


def get_labels(data_dir, latent_names):
    train_file = os.path.join(data_dir, "train.jsonl")
    train_data = [json.loads(line) for line in open(train_file)]
    id2labels = {}
    labels_set = set()
    for datum in train_data:
        labs = {key: val for (key, val) in datum.items()
                if key in latent_names}
        id2labels[datum["id"]] = labs
        labels_set.update(set(labs.keys()))
    return id2labels, labels_set


def train_lr(latent_name, id2z, label_name, id2labels):
    ordered_ids = list(id2z.keys())
    np.random.shuffle(ordered_ids)
    V = np.array([id2labels[uuid][label_name] for uuid in ordered_ids])
    Z = np.array([id2z[uuid] for uuid in ordered_ids])
    if len(Z.shape) == 1:
        Z = np.expand_dims(Z, axis=-1)
        plot_z_v(Z, V)
    Z = StandardScaler().fit_transform(Z)

    print(Counter(V))
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(Z, V)
    preds = clf.predict(Z)
    print(Counter(preds))
    return classification_report(V, preds)


def plot_z_v(Z, V):
    Z = Z.flatten()
    for v in set(V):
        zv = Z[V == v]
        sns.kdeplot(zv, label=str(v))
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
