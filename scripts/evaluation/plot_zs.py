import os
import re
import json
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--data_split", type=str,
                        choices=["train", "dev", "test"])
    parser.add_argument("--epoch", type=int, default=-1)
    return parser.parse_args()


def main(args):
    zs_dir = os.path.join(args.metadata_dir, 'z')
    if args.epoch == -1:
        epoch = get_last_epoch(zs_dir)
    else:
        epoch = args.epoch

    z_files = glob(os.path.join(zs_dir, f"{args.data_split}_*_{epoch}.log"))
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

    # Set up the subplots
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax_neg = fig.add_subplot(gs[0, 0])
    ax_neg.set_title("Negation", fontdict={"fontsize": 18})
    #ax_neg.set_xticks([])
    ax_neg.set_yticks([])
    ax_unc = fig.add_subplot(gs[0, 1])
    ax_unc.set_title("Uncertainty", fontdict={"fontsize": 18})
    #ax_unc.set_xticks([])
    ax_unc.set_yticks([])
    ax_con_neg = fig.add_subplot(gs[1, 0])
    ax_con_neg.set_aspect(1)
    ax_con_neg.set_title("Content - Negation", fontdict={"fontsize": 18})
    ax_con_neg.set_xticks([])
    ax_con_neg.set_yticks([])
    ax_con_unc = fig.add_subplot(gs[1, 1])
    ax_con_unc.set_aspect(1)
    ax_con_unc.set_title("Content - Uncertainty", fontdict={"fontsize": 18})
    ax_con_unc.set_xticks([])
    ax_con_unc.set_yticks([])

    for (latent_name, zfile) in zip(latent_names, z_files):
        zs = np.loadtxt(zfile, delimiter=',')

        if latent_name == "polarity":
            plot_negation(zs, Vs[latent_name], ax_neg)
        elif latent_name == "uncertainty":
            plot_uncertainty(zs, Vs[latent_name], ax_unc)
        elif latent_name == "content":
            plot_content(zs, Vs, ax_con_neg, variable="negation")
            plot_content(zs, Vs, ax_con_unc, variable="uncertainty")
    plt.show()


def plot_negation(zs, labels, axis):
    colors = {"positive": "#ef8a62", "negative": "#67a9cf"}
    for lab_val in set(labels):
        mask = np.array(labels) == lab_val
        sns.histplot(zs[mask], color=colors[lab_val], alpha=0.8,
                     ax=axis, label=lab_val, linewidth=0)
    axis.legend(fontsize=14)


def plot_uncertainty(zs, labels, axis):
    colors = {"certain": "#af8dc3", "uncertain": "#7fbf7b"}
    ci = 0
    for lab_val in set(labels):
        mask = np.array(labels) == lab_val
        sns.histplot(zs[mask], color=colors[lab_val], alpha=0.8,
                     ax=axis, label=lab_val, linewidth=0)
        ci += 1
    axis.legend(fontsize=14)


def plot_content_old(zs, labels_dict, axis):
    z_emb = TSNE(n_components=2).fit_transform(zs)

    df = pd.DataFrame({"z0": z_emb[:, 0], "z1": z_emb[:, 1],
                       "negation": labels_dict["polarity"],
                       "uncertainty": labels_dict["uncertainty"]})
    colors = ["#ef8a62", "#67a9cf"]
    sns.scatterplot(data=df, x="z0", y="z1", hue="negation", alpha=0.8,
                    style="uncertainty", palette=colors, ax=axis)


def plot_content(zs, labels_dict, axis, variable="negation"):
    z_emb = TSNE(n_components=2).fit_transform(zs)

    key = variable
    colors = {"certain": "#af8dc3", "uncertain": "#7fbf7b"}
    if variable == "negation":
        key = "polarity"
        colors = {"positive": "#ef8a62", "negative": "#67a9cf"}
    df = pd.DataFrame({"z0": z_emb[:, 0], "z1": z_emb[:, 1],
                       variable: labels_dict[key]})
    sns.scatterplot(data=df, x="z0", y="z1", hue=variable,
                    hue_order=labels_dict[key][::-1], palette=colors, ax=axis)
    axis.get_legend().remove()


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


if __name__ == "__main__":
    args = parse_args()
    main(args)
