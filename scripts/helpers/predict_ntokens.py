import os
import re
import json
import argparse
from glob import glob
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from vae import data_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["train", "dev", "test"])
    parser.add_argument("--latent_names", type=str, default=None,
                        nargs='+')
    return parser.parse_args()


def main(args):
    zs_dir = os.path.join(args.metadata_dir, 'z')
    epoch = get_last_epoch(zs_dir)
    z_files = glob(os.path.join(zs_dir, f"{args.dataset}_*_{epoch}.log"))
    if args.latent_names is None:
        latent_names = get_latent_names(z_files)
    else:
        latent_names = args.latent_names
    latent_2_zfile = dict(zip(latent_names, z_files))

    latent_name_combos = []
    for i in range(len(latent_names)):
        latent_name_combos.extend(combinations(latent_names, i+1))

    ids_dir = os.path.join(args.metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"{args.dataset}_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    id2labels = get_n_tokens(args.data_dir, args.dataset)
    y = np.array([id2labels[uuid] for uuid in ids])

    r2s = {}
    coefs = {}
    intercepts = {}
    for latent_names in tqdm(latent_name_combos):
        Z = None
        for name in latent_names:
            zfile = latent_2_zfile[name]
            zs = np.loadtxt(zfile, delimiter=',')
            if len(zs.shape) == 1:
                zs = zs.reshape(-1, 1)
            if Z is not None:
                Z = np.concatenate([Z, zs], axis=-1)
            else:
                Z = zs
        lr = LinearRegression().fit(Z, y)
        r2 = lr.score(Z, y)
        r2s[latent_names] = r2
        coefs[latent_names] = lr.coef_

        if ''.join(latent_names) == "content":
            print("CONTENT SPACE")
            print("Measuring R2 of each dimension...")
            max_coef_dims = np.argsort(lr.coef_)[::-1]  #[:3]
            coef_r2s = []
            for (coef_dim, coef) in enumerate(lr.coef_):
                zcoef = Z[:, coef_dim].reshape(-1, 1)
                lr_coef = LinearRegression().fit(zcoef, y)
                coef_r2 = lr_coef.score(zcoef, y)
                coef_r2s.append(coef_r2)
            sorted_dims_r2s = list(sorted(
                enumerate(coef_r2s), key=lambda x: x[1], reverse=True))
            print(f"{'dim':<5}: R2")
            for (dim, r2) in sorted_dims_r2s:
                print(f"{dim:<5}: {r2:<7.4f}")
                # plt.scatter(Z[:, dim], y)
                # plt.title(f"({dim}) R2: {r2:.4f}")
                # plt.show()

        intercepts[latent_names] = lr.intercept_

    print("RESULTS")
    for (names, r2) in r2s.items():
        name_str = '+'.join(names)
        print(f"{name_str}: R2 = {r2:.4f}")
        sorted_coefs = sorted(enumerate(coefs[names]), key=lambda x: x[1],
                              reverse=True)
        print(f"  highest (dim, coef): {sorted_coefs[:3]}")
        print(f"  intercept: {intercepts[names]}")


def get_last_epoch(directory):
    files = os.listdir(directory)
    epochs = {int(re.findall(r'.*_([0-9]+)\.log', fname)[0])
              for fname in files}
    return max(epochs)


def get_latent_names(filenames):
    latent_names = []
    for fname in filenames:
        name = re.findall(r'.*?_(\w+)_[0-9]+.log', fname)[0]
        latent_names.append(name)
    return latent_names


def get_n_tokens(data_dir, dataset):
    data_file = os.path.join(data_dir, f"{dataset}.jsonl")
    data = [json.loads(line) for line in open(data_file)]
    id2labels = {}
    for datum in data:
        try:
            lab = datum["n_tokens"]
        except KeyError:
            toks = data_utils.preprocess_sentences([datum["sentence"]])[0]
            lab = len(toks)
        id2labels[datum["id"]] = lab
    return id2labels


if __name__ == "__main__":
    args = parse_args()
    main(args)
