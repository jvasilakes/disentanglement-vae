import os
import re
import json
import argparse
from glob import glob
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import torch.distributions as D
from tqdm import trange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("data_split", type=str,
                        choices=["train", "dev", "test"])
    parser.add_argument("--num_resamples", type=int, default=10)
    return parser.parse_args()


def main(args):
    zs_dir = os.path.join(args.metadata_dir, 'z')
    epoch = get_last_epoch(zs_dir)

    z_files = glob(os.path.join(zs_dir, f"{args.data_split}_*_{epoch}.log"))
    mu_files = glob(os.path.join(args.metadata_dir, "mu",
                                 f"{args.data_split}_*_{epoch}.log"))
    logvar_files = glob(os.path.join(args.metadata_dir, "logvar",
                                     f"{args.data_split}_*_{epoch}.log"))
    latent_names = get_latent_names(z_files)

    ids_dir = os.path.join(args.metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"{args.data_split}_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    # id2labels = {uuid: {latent_name: value for latent_name in latent_names}}
    # But this will not include the "content" latent.
    # labels_set = {lname for lname in latent_names if lname is a supervised latent}  # noqa
    id2labels, labels_set = get_labels(
        args.data_dir, args.data_split, latent_names)
    Vs = defaultdict(list)
    for uuid in ids:
        labels = id2labels[uuid]
        for (lab_name, val) in labels.items():
            Vs[lab_name].append(val)

    rows = []
    zipped = list(zip(latent_names, z_files, mu_files, logvar_files))
    for i in trange(args.num_resamples):
        for (latent_name, zfile, mufile, logvarfile) in zipped:
            for lab_name in labels_set:
                if lab_name == latent_name:
                    continue
                # Predict lab_name from z_latent_name
                mus = np.loadtxt(mufile, delimiter=',')
                logvars = np.loadtxt(logvarfile, delimiter=',')
                # zs = np.loadtxt(zfile, delimiter=',')
                zs = sample_from_latent(mus, logvars).numpy()
                # print(f"{latent_name} |-> {lab_name}")

                label_values = set(Vs[lab_name])
                zs_per_label_val = {}
                for label_val in label_values:
                    mask = np.array(Vs[lab_name]) == label_val
                    zs_vals = zs[mask]
                    zs_per_label_val[label_val] = zs_vals
                    row = [i, latent_name, lab_name, label_val,
                           zs_vals.mean(), zs_vals.std()]
                    rows.append(row)

    colnames = ["sample_num", "latent", "label",
                "label_val", "z_mean", "z_std"]
    df = pd.DataFrame(rows, columns=colnames)
    summarize(df)


def summarize(df):
    grouped = df.groupby(["latent", "label", "label_val"])
    summ = grouped.agg("mean").drop("sample_num", axis="columns")
    diffs = summ.groupby(["latent", "label"]).diff()
    diffs = diffs.droplevel("label_val").dropna(axis=0, how="all").abs()
    diffs.columns = ["z_mean_diff", "z_std_diff"]
    print("""For each latent, measures the absolute difference of the mean
    and standard deviation of the z values between the different
    values of label.""")
    print(diffs)


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


def get_labels(data_dir, data_split, latent_names):
    data_file = os.path.join(data_dir, f"{data_split}.jsonl")
    data = [json.loads(line) for line in open(data_file)]
    id2labels = {}
    labels_set = set()
    for datum in data:
        labs = {key: val for (key, val) in datum.items()
                if key in latent_names}
        id2labels[datum["id"]] = labs
        labels_set.update(set(labs.keys()))
    return id2labels, labels_set


def sample_from_latent(mus, logvars):
    mus = torch.tensor(mus).float()
    logvars = torch.tensor(logvars).float()
    if len(mus.size()) == 1:
        dist = D.Normal(mus, logvars.exp())
    else:
        vars_mat = to_diagonal_matrix(logvars.exp())
        dist = D.MultivariateNormal(mus, vars_mat)
    zs = dist.sample()
    return zs


def to_diagonal_matrix(tensor):
    N, M = tensor.size()
    mat = torch.eye(M).repeat(N, 1, 1)
    mask = mat.bool()
    mat[mask] = tensor.flatten()
    return mat


if __name__ == "__main__":
    args = parse_args()
    main(args)
