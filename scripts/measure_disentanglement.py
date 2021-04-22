import os
import re
import json
import argparse
from glob import glob
from collections import Counter, defaultdict

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.distributions as D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("metadata_dir", type=str,
                        help="Directory containing z/ and ordered_ids/")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing {train,dev,test.jsonl}.")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Which training epoch to run on. Default last.")
    parser.add_argument("--num_resamples", type=int, default=10)
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
    mu_files = glob(os.path.join(args.metadata_dir, "mu",
                                 f"train_*_{epoch}.log"))
    logvar_files = glob(os.path.join(args.metadata_dir, "logvar",
                                     f"train_*_{epoch}.log"))
    latent_names = get_latent_names(z_files)

    ids_dir = os.path.join(args.metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"train_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    # id2labels = {uuid: {latent_name: value for latent_name in latent_names}}
    # But this will not include the "content" latent.
    # labels_set = {lname for lname in latent_names if lname is a supervised latent}  # noqa
    id2labels, labels_set = get_labels(args.data_dir, latent_names)
    Vs = defaultdict(list)
    for uuid in ids:
        labels = id2labels[uuid]
        for (lab_name, val) in labels.items():
            Vs[lab_name].append(val)

    mis = defaultdict(dict)
    Hvs = {}
    zipped = list(zip(latent_names, z_files, mu_files, logvar_files))
    for i in range(args.num_resamples):
        for (latent_name, zfile, mufile, logvarfile) in zipped:
            for lab_name in labels_set:
                # Predict lab_name from z_latent_name
                mus = np.loadtxt(mufile, delimiter=',')
                logvars = np.loadtxt(logvarfile, delimiter=',')
                #zs = np.loadtxt(zfile, delimiter=',')
                zs = sample_from_latent(mus, logvars).numpy()
                id2z = dict(zip(ids, zs))
                print(f"{latent_name} |-> {lab_name}")
                model, results = train_lr(latent_name, id2z, lab_name, id2labels)
                print(results)

                if lab_name not in Hvs.keys():
                    Hvs[lab_name] = compute_entropy_freq(Vs[lab_name])

                mi = compute_mi(zs, Vs[lab_name])
                print("MI: ", mi)
                print()
                mis[lab_name][latent_name] = mi
        # TODO: Log MIGs for each resample as JSONL, one line per resample.
        # {latent_name: {MIG: float, MIs: [float, float], Hv: float}, ...}
        migs = compute_migs(mis, Hvs)
        for (key, val) in migs.items():
            print(f"{key}: {val}")
        print("-----------------------------------")


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
        # plot_z_v(Z, V)
    Z = StandardScaler().fit_transform(Z)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(Z, V)
    preds = clf.predict(Z)
    return clf, classification_report(V, preds)


def plot_z_v(Z, V):
    Z = Z.flatten()
    for v in set(V):
        zv = Z[V == v]
        sns.kdeplot(zv, label=str(v))
    plt.show()


def to_diagonal_matrix(tensor):
    N, M = tensor.size()
    mat = torch.eye(M).repeat(N, 1, 1)
    mask = mat.bool()
    mat[mask] = tensor.flatten()
    return mat


def compute_probabilities_latents(mus, logvars, zs):
    mus = torch.tensor(mus).float()
    logvars = torch.tensor(logvars).float()
    zs = torch.tensor(zs).float()
    if len(mus.size()) == 1:
        dist = D.Normal(mus, logvars.exp())
    else:
        vars_mat = to_diagonal_matrix(logvars.exp())
        dist = D.MultivariateNormal(mus, vars_mat)
    logprobs = dist.log_prob(zs)
    probs = logprobs.exp()
    return probs


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


def compute_entropy_freq(xs, mean=True):
    xs = np.array(xs)
    counts = Counter(xs)
    probs = np.array([counts[x]/len(xs) for x in xs])
    if mean is True:
        probs = [np.mean(probs[xs == x]) for x in set(xs)]
    else:
        probs = probs / np.sum(probs)
    H = -np.sum(probs * np.log(probs))
    return H


def compute_entropy_oracle(xs):
    df = pd.DataFrame(xs)
    probs = df.value_counts().div(len(df))
    H = -np.sum(probs.values * np.log(probs.values))
    return H


def compute_joint_entropy_oracle(zs, vs):
    zs = zs.flatten()
    df = pd.DataFrame(np.array(list(zip(zs, vs))))
    probs = df.groupby(list(df.columns)).size().div(len(df))
    H = -np.sum(probs.values * np.log(probs.values))
    return H


def compute_mi(zs, vs, discrete_z=False):
    if len(zs.shape) == 1:
        zs = zs.reshape(-1, 1)
    MIs = mutual_info_classif(zs, vs, discrete_features=discrete_z)
    return MIs.mean()


def compute_migs(mi_dict, Hvs):
    migs = defaultdict(dict)
    for lab_name in mi_dict.keys():
        lab_mis = []
        latent_names = []
        for latent_name in mi_dict[lab_name].keys():
            mi = mi_dict[lab_name][latent_name]
            lab_mis.append(mi)
            latent_names.append(latent_name)
        sorted_pairs = sorted(zip(lab_mis, latent_names),
                              key=lambda x: x[0], reverse=True)
        sorted_lab_mis, sorted_names = zip(*sorted_pairs)
        Hv = Hvs[lab_name]
        mig_v = (sorted_lab_mis[0] - sorted_lab_mis[1]) / Hv
        migs[lab_name][sorted_names[:2]] = (mig_v, (sorted_lab_mis[:2], Hv))
    return migs


# ===========================
#    Testing functions
# ===========================

def test_random(N):
    zs = np.random.randn(N).reshape(-1, 1)
    vs = np.random.binomial(1, 0.5, size=N)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_kinda_predictive(N):
    zs = np.random.randn(N).reshape(-1, 1)
    vs = np.array([0 if z < 0.0 else 1 for z in zs])
    idxs = np.random.randint(0, len(vs), size=N-5000)
    vs[idxs] = np.logical_not(vs[idxs]).astype(int)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_predictive(N):
    zs = np.random.randn(N).reshape(-1, 1)
    vs = [0 if z < 0.0 else 1 for z in zs]
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_bijective(N, predictive=False):
    zs = np.random.binomial(1, 0.5, N)
    if predictive is True:
        vs = [0 if z == 1 else 1 for z in zs]
    else:
        vs = np.random.binomial(1, 0.5, N)
    zs = zs.reshape(-1, 1)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=True)
    print("MI: ", MI)


def test_bijective_oracle(N, predictive=False):
    zs = np.random.binomial(1, 0.5, N)
    if predictive is True:
        vs = [0 if z == 1 else 1 for z in zs]
    else:
        vs = np.random.binomial(1, 0.5, N)
    Hz = compute_entropy_oracle(zs)
    Hv = compute_entropy_oracle(vs)
    Hvz = compute_joint_entropy_oracle(zs, vs)
    print("H[z]: ", Hz)
    print("H[v]: ", Hv)
    print("H[v,z]: ", Hvz)
    MI_joint = Hv + Hz - Hvz
    print("MI = H[z] + H[v] - H[v,z]: ", MI_joint)


if __name__ == "__main__":
    args = parse_args()
    if args.test is True:
        N = 1000000
        print("RANDOM")
        test_random(N)
        print("KINDA PREDICTIVE")
        test_kinda_predictive(N)
        print("PREDICTIVE")
        test_predictive(N)
        print()
        print("BIJECTIVE ORACLE")
        print("  random")
        test_bijective_oracle(N)
        print("  predictive")
        test_bijective_oracle(N, predictive=True)
        print()
        print("BIJECTIVE")
        print("  random")
        test_bijective(N)
        print("  predictive")
        test_bijective(N, predictive=True)
        print()
    else:
        main(args)
