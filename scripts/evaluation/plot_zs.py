import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # noqa
from glob import glob
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("zs_dir", type=str)
    parser.add_argument("data_split", type=str,
                        choices=["train", "dev", "test"])
    parser.add_argument("latent_name", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("-E", "--skip_epochs", type=int, default=1,
                        help="Plot every E epochs")
    return parser.parse_args()


def main(args):
    outdir = os.path.join(args.outdir, args.data_split)
    os.makedirs(outdir, exist_ok=True)
    z_files = glob(os.path.join(
        args.zs_dir, f"{args.data_split}_{args.latent_name}*"))
    if z_files == []:
        raise ValueError(f"No z files found at '{args.zs_dir}/*_{args.data_split}_{args.latent_name}'")  # noqa

    dfs = defaultdict(pd.DataFrame)
    for zf in z_files:
        epoch = int(zf.split('_')[-1].replace(".log", ''))
        if epoch % args.skip_epochs != 0:
            continue
        data = pd.read_csv(zf, header=None)
        for dim in data.columns:
            dfs[dim][epoch] = data[dim]

    for (dim, df) in dfs.items():
        df = df[df.columns.sort_values(ascending=False)]
        df.loc[:, "N(0,1)"] = np.random.randn(len(df))
        plot = sns.displot(data=df, kind="kde", linewidth=2, palette="flare")
        norm_line = plot.ax.get_lines()[0]
        norm_line.set_color("#cc0000")
        norm_line.set_linestyle("--")
        norm_leg = plot.legend.get_lines()[-1]
        norm_leg.set_color("#cc0000")
        norm_leg.set_linestyle("--")
        plot.savefig(
                os.path.join(outdir, f"zs_{args.latent_name}_dim{dim}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
