import json
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"xtick.labelsize": 14})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("MIG_files", type=str, nargs='+',
                        help="MIG*.jsonl files to plot.")
    parser.add_argument("outfile", type=str,
                        help="Where to save the plot.")
    parser.add_argument("--model_names", type=str, nargs='+', required=True,
                        help="Name to use for each MIG file.")
    return parser.parse_args()


def plot_migs(args):
    all_mi_dfs = []
    all_mig_dfs = []
    for (model_name, mig_file) in zip(args.model_names, args.MIG_files):
        data = [json.loads(line) for line in open(mig_file)]
        mi_df = get_mi_dataframe(data)
        mig_df = get_mig_dataframe(data)
        all_mi_dfs.append(mi_df)
        all_mig_dfs.append(mig_df)
    plot = plot_mis_migs(all_mi_dfs, all_mig_dfs, args.model_names)
    plt.show()
    # plt.savefig(args.outfile)


def get_mi_dataframe(mig_data):
    df_rows = []
    for (i, datum) in enumerate(mig_data):
        for label_name in datum.keys():
            if label_name == "sample_num":
                continue
            mis = datum[label_name]["sorted_MIs"]
            names = datum[label_name]["sorted_latents"]
            for (latent_name, latent_mi) in zip(names, mis):
                if latent_name == "polarity":
                    latent_name = "negation"
                if label_name == "polarity":
                    label_name = "negation"
                df_rows.append({"sample_num": i, "label_name": label_name,
                                "latent_name": latent_name, "MI": latent_mi})
    return pd.DataFrame(df_rows)


def get_mig_dataframe(mig_data):
    df_rows = defaultdict(list)
    for (i, datum) in enumerate(mig_data):
        for label_name in datum.keys():
            if label_name == "sample_num":
                continue
            colname = label_name
            if label_name == "polarity":
                colname = "negation"
            df_rows[colname].append(datum[label_name]["MIG"])
    return pd.DataFrame(df_rows)


def plot_mis_migs(mi_dfs, mig_dfs, names):
    num_subplots = len(mi_dfs)
    fig, axs = plt.subplots(2, num_subplots)

    colors = ["#ef8a62", "#67a9cf"]

    i = 0
    for (mig_df, model_name) in zip(mig_dfs, names):
        box = mig_df.boxplot(column=sorted(mig_df.columns), ax=axs[0, i],
                             patch_artist=True, return_type="dict",
                             widths=0.75)
        for (patch, color) in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        axs[0, i].set_title(model_name, fontsize=16)
        axs[0, i].set_ylim(0.0, 0.8)
        axs[0, i].set_xticklabels(["Neg", "Unc"])
        if i == 0:
            axs[0, i].set_ylabel("MIG", fontsize=14)
        if i > 0:
            axs[0, i].axes.yaxis.set_ticklabels([])
        i += 1

    i = 0
    for (mi_df, model_name) in zip(mi_dfs, names):
        mi_summ = mi_df.groupby(["label_name", "latent_name"]).agg(
            "mean").drop("sample_num", axis="columns")
        mi_summ = mi_summ.unstack(level=0)
        # Flatten the multi-index
        mi_summ.columns = [idx[1] for idx in mi_summ.columns]

        yerr = mi_df.groupby(["label_name", "latent_name"]).agg(
            "std").drop("sample_num", axis="columns")
        yerr = yerr.unstack(level=0)
        # Flatten the multi-index
        yerr.columns = [idx[1] for idx in yerr.columns]
        mi_summ.plot.bar(ax=axs[1, i], color=colors, rot=0, legend=False,
                         yerr=yerr)
        xlabs = [name[0].upper() for name in mi_summ.index]
        axs[1, i].axes.xaxis.set_ticklabels(xlabs)
        axs[1, i].set_xlabel('')
        if i == 0:
            axs[1, i].set_ylabel("MI\n(b/n latent and label)", fontsize=13)
        if i > 0:
            axs[1, i].axes.yaxis.set_ticklabels([])
        i += 1

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args()
    plot_migs(args)
