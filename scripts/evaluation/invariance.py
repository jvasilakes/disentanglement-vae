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
from matplotlib import cm

import seaborn as sns
import matplotlib.pyplot as plt


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
    print(f"Generative factors: {labels_set}")
    if len(labels_set) <= 1:
        msg = "This script requires at least two generative factors"
        raise ValueError(msg)

    Vs = defaultdict(list)
    for uuid in ids:
        labels = id2labels[uuid]
        for (lab_name, val) in labels.items():
            Vs[lab_name].append(val)

    rows = []
    zs_log = defaultdict(dict)
    zipped = list(zip(latent_names, z_files, mu_files, logvar_files))
    for i in trange(args.num_resamples):
        for (latent_name, zfile, mufile, logvarfile) in zipped:
            for vary_label in labels_set:
                static_label = latent_name
                if vary_label == static_label:
                    continue
                if static_label == "content":
                    continue

                # Predict lab_name from z_latent_name
                mus = np.loadtxt(mufile, delimiter=',')
                logvars = np.loadtxt(logvarfile, delimiter=',')
                # zs = np.loadtxt(zfile, delimiter=',')
                zs = sample_from_latent(mus, logvars).numpy()
                # print(f"{latent_name} |-> {lab_name}")

                for static_label_val in set(Vs[static_label]):
                    static_mask = np.array(Vs[static_label]) == static_label_val  # noqa
                    for (ci, vary_label_val) in enumerate(set(Vs[vary_label])):
                        vary_mask = np.array(Vs[vary_label]) == vary_label_val
                        mask = np.logical_and(static_mask, vary_mask)
                        zs_vals = zs[mask]

                        try:
                            zs_log[static_label_val][vary_label][vary_label_val] = zs_vals  # noqa
                        except KeyError:
                            zs_log[static_label_val][vary_label] = {}
                            zs_log[static_label_val][vary_label][vary_label_val] = zs_vals  # noqa

                        row = [i, latent_name, static_label, static_label_val,
                               vary_label, vary_label_val, zs_vals.mean(),
                               zs_vals.std()]
                        rows.append(row)

    colnames = ["sample_num", "latent", "static_label", "static_label_val",
                "vary_label", "vary_label_val", "z_mean", "z_std"]
    df = pd.DataFrame(rows, columns=colnames)
    summarize(df)
    make_plot(df)


def summarize(df):
    grouped = df.groupby(["latent", "static_label", "static_label_val",
                          "vary_label", "vary_label_val"])
    summ = grouped.agg("mean").drop("sample_num", axis="columns")
    diffs = summ.groupby(["latent", "static_label", "static_label_val",
                          "vary_label"]).diff()
    diffs = diffs.droplevel("vary_label_val").dropna(axis=0, how="all").abs()
    diffs.columns = ["z_mean_diff", "z_std_diff"]
    print(diffs)


def make_plot(df):
    label_combos_df = df[["static_label", "static_label_val"]].value_counts()
    label_combos_df = label_combos_df.reset_index().drop(columns=0)
    label_group_sizes = label_combos_df.groupby("static_label").size()
    rows = len(label_group_sizes)
    cols = max(label_group_sizes) * (rows - 1)
    fig, axs = plt.subplots(rows, cols)

    cmaps = ['Accent', 'Set1', 'Set2', 'Set3', 'tab10']

    # update row when we change the static label
    # update the col when we change the vary label
    #    set it back to 0 when we switch rows
    row = col = 0  # subplot index
    cmap_lookup = {}
    current_static_lab = None
    static_gb = df.groupby(["static_label", "static_label_val"])
    for (static_group, _) in static_gb:
        if current_static_lab is None:
            current_static_lab = static_group[0]
        if static_group[0] != current_static_lab:
            current_static_lab = static_group[0]
            row += 1
            col = 0
        static_sub_gb = static_gb.get_group(static_group)

        current_vary_lab = None
        vary_gb = static_sub_gb.groupby(["vary_label", "vary_label_val"])
        for (ci, (vary_group, _)) in enumerate(vary_gb):
            if current_vary_lab is None:
                current_vary_lab = vary_group[0]
            if vary_group[0] != current_vary_lab:
                current_vary_lab = vary_group[0]
                col += 1
            try:
                cmap_i = cmap_lookup[current_vary_lab]
            except KeyError:
                cmap_lookup[current_vary_lab] = len(cmap_lookup)
                cmap_i = cmap_lookup[current_vary_lab]
            vary_sub_gb = vary_gb.get_group(vary_group)
            mus = vary_sub_gb["z_mean"].to_numpy()
            logvars = vary_sub_gb["z_std"].to_numpy()
            zs = sample_from_latent(mus, logvars).numpy()
            lab = '='.join(str(item) for item in vary_group)
            cmap = cm.get_cmap(cmaps[cmap_i % len(cmaps)])
            sns.kdeplot(zs, alpha=0.5, fill=True, label=lab, ax=axs[row, col],
                        color=cmap(ci))
            axs[row, col].legend()
            title = '='.join(str(item) for item in static_group)
            axs[row, col].set_title(title, fontsize=16)
        col += 1

    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel('')

    fig.subplots_adjust(wspace=0.1)
    plt.show()


def get_last_epoch(directory):
    files = os.listdir(directory)
    epochs = {int(re.findall(r'.*_([0-9]+)\.log', fname)[0])
              for fname in files}
    return max(epochs)


def get_latent_names(filenames):
    latent_names = []
    for fname in filenames:
        name = re.findall(r'[train|dev|test]_([\w_]+)_[0-9]+.log', fname)[0]
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
