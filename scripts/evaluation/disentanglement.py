import os
import re
import csv
import json
import argparse
from glob import glob
from tqdm import trange
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
from sklearn.metrics import precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Specify test, compute, or summarize")

    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(test=True, compute=False, summarize=False)
    test_parser.add_argument("-N", type=int, default=100000,
                             help="Number of samples to use.")
    test_parser.add_argument("-K", type=int, default=2,
                             help="Number of classes.")
    test_parser.add_argument("--n_features", type=int, default=1,
                             help="Number of features.")

    compute_parser = subparsers.add_parser("compute")
    compute_parser.set_defaults(test=False, compute=True, summarize=False)
    compute_parser.add_argument("metadata_dir", type=str,
                                help="""Directory containing z/
                                        and ordered_ids/""")
    compute_parser.add_argument("data_dir", type=str,
                                help="""Directory containing
                                        {train,dev,test.jsonl}.""")
    compute_parser.add_argument("dataset", type=str,
                                choices=["train", "dev", "test"],
                                help="Dataset to run on.")
    compute_parser.add_argument("outdir", type=str,
                                help="Where to save the results.")
    compute_parser.add_argument("--epoch", type=int, default=-1,
                                help="""Which epoch to run on.
                                        Default last.""")
    compute_parser.add_argument("--num_resamples", type=int, default=10,
                                help="""The number of times to
                                        resample and compute MIG""")

    summ_parser = subparsers.add_parser("summarize")
    summ_parser.set_defaults(test=False, compute=False, summarize=True)
    summ_parser.add_argument("dataset", type=str,
                             choices=["train", "dev", "test"],
                             help="Dataset to run on.")
    summ_parser.add_argument("outdir", type=str,
                             help="Directory containing results to summarize.")

    args = parser.parse_args()
    if [args.test, args.compute, args.summarize] == [False, False, False]:
        parser.print_help()
    return args


def compute(args, model=None):
    """
    Get zs from metadata_dir/z/dataset_*_epoch.log
    Get labels from data_dir/dataset.jsonl using ids from
        metadata_dir/ordered_ids/dataset_epoch.log
    For each zs/labels pair:
        Estimate logistic regression model from zs to labels.
        Evaluate above model on data_dir/{train,dev,test}.jsonl
    """
    os.makedirs(args.outdir, exist_ok=True)

    zs_dir = os.path.join(args.metadata_dir, 'z')
    if args.epoch == -1:
        epoch = get_last_epoch(zs_dir)
    else:
        epoch = args.epoch

    z_files = glob(os.path.join(zs_dir, f"{args.dataset}_*_{epoch}.log"))
    mu_files = glob(os.path.join(args.metadata_dir, "mu",
                                 f"{args.dataset}_*_{epoch}.log"))
    logvar_files = glob(os.path.join(args.metadata_dir, "logvar",
                                     f"{args.dataset}_*_{epoch}.log"))
    latent_names = get_latent_names(z_files)

    ids_dir = os.path.join(args.metadata_dir, "ordered_ids")
    ids_file = os.path.join(ids_dir, f"{args.dataset}_{epoch}.log")
    ids = [uuid.strip() for uuid in open(ids_file)]

    # id2labels = {uuid: {latent_name: value for latent_name in latent_names}}
    # But this will not include the "content" latent.
    # labels_set = {lname for lname in latent_names if lname is a supervised latent}  # noqa
    id2labels, labels_set = get_labels(
        args.data_dir, args.dataset, latent_names)
    Vs = defaultdict(list)
    for uuid in ids:
        labels = id2labels[uuid]
        for (lab_name, val) in labels.items():
            Vs[lab_name].append(val)

    migs_outfile = os.path.join(args.outdir, f"MIGS_{args.dataset}.jsonl")
    preds_outfile = os.path.join(args.outdir,
                                 f"predictions_{args.dataset}.csv")
    zipped = list(zip(latent_names, z_files, mu_files, logvar_files))
    Hvs = {}
    for i in trange(args.num_resamples):
        mis = defaultdict(dict)
        pred_results = []
        for (latent_name, zfile, mufile, logvarfile) in zipped:
            for lab_name in labels_set:
                # Predict lab_name from z_latent_name
                mus = np.loadtxt(mufile, delimiter=',')
                logvars = np.loadtxt(logvarfile, delimiter=',')
                # zs = np.loadtxt(zfile, delimiter=',')
                zs = sample_from_latent(mus, logvars).numpy()
                id2z = dict(zip(ids, zs))
                # print(f"{latent_name} |-> {lab_name}")
                # Precision, recall, f1
                model, (p, r, f, _) = train_lr(latent_name, id2z,
                                               lab_name, id2labels,
                                               random_state=i)
                pred_results.append([i, latent_name, lab_name, p, r, f])

                if lab_name not in Hvs.keys():
                    Hvs[lab_name] = compute_entropy_freq(Vs[lab_name])

                mi = compute_mi(zs, Vs[lab_name])
                mis[lab_name][latent_name] = mi
        migs = compute_migs(mis, Hvs)
        with open(migs_outfile, 'a') as outF:
            migs["sample_num"] = i
            json.dump(migs, outF)
            outF.write('\n')
        with open(preds_outfile, 'a') as outF:
            writer = csv.writer(outF, delimiter=',')
            if i == 0:
                writer.writerow(["sample_num", "latent_name", "label_name",
                                 "precision", "recall", "F1"])
            for line in pred_results:
                writer.writerow(line)


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


def get_labels(data_dir, dataset, latent_names):
    data_file = os.path.join(data_dir, f"{dataset}.jsonl")
    data = [json.loads(line) for line in open(data_file)]
    id2labels = {}
    labels_set = set()
    for datum in data:
        labs = {key: val for (key, val) in datum.items()
                if key in latent_names}
        id2labels[datum["id"]] = labs
        labels_set.update(set(labs.keys()))
    return id2labels, labels_set


def train_lr(latent_name, id2z, label_name, id2labels, random_state=0):
    ordered_ids = list(id2z.keys())
    np.random.shuffle(ordered_ids)
    V = np.array([id2labels[uuid][label_name] for uuid in ordered_ids])
    Z = np.array([id2z[uuid] for uuid in ordered_ids])
    if len(Z.shape) == 1:
        Z = np.expand_dims(Z, axis=-1)
        # plot_z_v(Z, V, title=f"{latent_name} |-> {label_name}")
    Z = StandardScaler().fit_transform(Z)
    clf = LogisticRegression(random_state=random_state,
                             class_weight="balanced",
                             penalty="none").fit(Z, V)
    preds = clf.predict(Z)
    return clf, precision_recall_fscore_support(V, preds, average="macro")


def plot_z_v(Z, V, title=""):
    Z = Z.flatten()
    for v in set(V):
        zv = Z[V == v]
        ax = sns.kdeplot(zv, label=str(v))
    ax.set_title(title)
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
    vs = vs.reshape(-1, 1)
    df = pd.DataFrame(np.concatenate((zs, vs), axis=1))
    probs = df.groupby(list(df.columns)).size().div(len(df))
    H = -np.sum(probs.values * np.log(probs.values))
    return H


def compute_mi(zs, vs, discrete_z=False):
    if len(zs.shape) == 1:
        zs = zs.reshape(-1, 1)
    MIs = mutual_info_classif(zs, vs, discrete_features=discrete_z)
    # Necessary to convert to regular float for JSON serialization
    return float(MIs.sum())


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
        migs[lab_name] = {"sorted_latents": sorted_names,
                          "MIG": mig_v,
                          "sorted_MIs": sorted_lab_mis,
                          "label_entropy": Hv}
    return migs


# ===========================
#    Testing functions
# ===========================

def test_random(N, K, n_features=1):
    zs = np.random.randn(N, n_features)
    if K < 2:
        raise ValueError("K must be >1")
    elif K == 2:
        vs = np.random.binomial(1, 0.5, size=N)
    else:
        vs = np.random.dirichlet([0.5]*K, size=N).argmax(axis=1)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_kinda_predictive(N, K, n_features=1):
    if K < 2:
        raise ValueError("K must be >1")
    zs = np.random.uniform(-K, K, size=(N, n_features))

    # For simplicitly, only use the first dimension for classification
    zs_d0 = zs[:, 0]
    stepsize = (zs_d0.max() - zs_d0.min()) / K
    thresholds = [zs_d0.min() + stepsize * (i+1) for i in range(K)]
    thresholds[-1] = zs_d0.max()
    vs = []
    for z in zs_d0:
        for i in range(K):
            if z <= thresholds[i]:
                break
        vs.append(i)
    vs = np.array(vs)

    idxs = np.random.randint(0, len(vs), size=int(N//5))
    vs[idxs] = np.random.randint(0, K, size=int(N//5))
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_predictive(N, K, n_features=1):
    if K < 2:
        raise ValueError("K must be >1")
    zs = np.random.uniform(-K, K, size=(N, n_features))
    zs_d0 = zs[:, 0]
    stepsize = (zs_d0.max() - zs_d0.min()) / K
    thresholds = [zs_d0.min() + stepsize * (i+1) for i in range(K)]
    thresholds[-1] = zs_d0.max()
    vs = []
    for z in zs_d0:
        for i in range(K):
            if z <= thresholds[i]:
                break
        vs.append(i)
    vs = np.array(vs)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=False)
    print("MI: ", MI)


def test_bijective(N, K, n_features=1, predictive=False):
    if K < 2:
        raise ValueError("K must be >1")
    vs = np.random.randint(0, K, size=N)
    if predictive is True:
        zs = vs
    else:
        zs = np.random.randint(0, K, size=N)
    zs = zs.reshape(-1, 1)
    clf = LogisticRegression(random_state=10, class_weight="balanced",
                             penalty="none").fit(zs, vs)
    print("LR accuracy: ", clf.score(zs, vs))
    Hv = compute_entropy_freq(vs)
    print("H[v]: ", Hv)
    MI = compute_mi(zs, vs, discrete_z=True)
    print("MI: ", MI)


def test_bijective_oracle(N, K, n_features=1, predictive=False):
    if K < 2:
        raise ValueError("K must be >1")
    vs = np.random.randint(0, K, size=N)
    if predictive is True:
        zs = vs.reshape(-1, 1).repeat(n_features, axis=1)
    else:
        zs = np.random.randint(0, K, size=(N, n_features))
    Hz = compute_entropy_oracle(zs)
    Hv = compute_entropy_oracle(vs)
    Hvz = compute_joint_entropy_oracle(zs, vs)
    print("H[z]: ", Hz)
    print("H[v]: ", Hv)
    print("H[v,z]: ", Hvz)
    MI_joint = Hv + Hz - Hvz
    print("MI = H[z] + H[v] - H[v,z]: ", MI_joint)


# ===========================
#    Summarization functions
# ===========================

def summarize_results(args):
    print(f"Summarizing results from {args.outdir}/*_{args.dataset}")
    print()

    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    migs_outfile = os.path.join(args.outdir, f"MIGS_{args.dataset}.jsonl")
    preds_outfile = os.path.join(
        args.outdir, f"predictions_{args.dataset}.csv")
    migs_data = [json.loads(line) for line in open(migs_outfile)]
    plot = summarize_migs(migs_data)
    plot_outfile = os.path.join(plot_dir, f"disentanglement_{args.dataset}")
    plot.savefig(f"{plot_outfile}.png", dpi=300)
    plot.savefig(f"{plot_outfile}.pdf", dpi=300)

    preds_data = pd.read_csv(preds_outfile)
    plot = summarize_preds(preds_data)
    plot_outfile = os.path.join(plot_dir, f"predictions_{args.dataset}")
    plot.savefig(f"{plot_outfile}.png", dpi=300)
    plot.savefig(f"{plot_outfile}.pdf", dpi=300)


def summarize_migs(migs_data):
    migs = defaultdict(list)
    mis = defaultdict(lambda: defaultdict(list))
    mis_rows = []
    for (i, datum) in enumerate(migs_data):
        for label_name in datum.keys():
            if label_name == "sample_num":
                continue
            migs[label_name].append(datum[label_name]["MIG"])
            mi = datum[label_name]["sorted_MIs"]
            names = datum[label_name]["sorted_latents"]
            for (latent_name, latent_mi) in zip(names, mi):
                mis[label_name][latent_name].append(latent_mi)
                mis_rows.append({"sample_num": i, "label_name": label_name,
                                 "latent_name": latent_name, "MI": latent_mi})

    fig, axs = plt.subplots(1, 2, figsize=[8, 6])

    mi_df = pd.DataFrame(mis_rows)
    print("======== MIs ========")
    mi_summ_df = mi_df.groupby(["label_name", "latent_name"]).agg(
        ["mean", "std"]).drop("sample_num", axis="columns")
    print(mi_summ_df.to_string())
    print()
    # Boxplot of the MIs
    mi_df.boxplot(column=["MI"], by=["label_name", "latent_name"],
                  ax=axs[0], rot=60)
    axs[0].set_title("MI per (label, latent) pair")

    migs_df = pd.DataFrame(migs)
    print("======== MIGs ========")
    migs_summ_df = migs_df.agg(["mean", "std", "size"]).T
    migs_summ_df.reset_index(inplace=True)
    migs_summ_df.columns = ["latent", "mean", "sd", "N"]
    print(migs_summ_df.to_string())
    print()
    # Boxplot of the MIGs
    migs_df.boxplot(column=sorted(migs_df.columns), ax=axs[1])
    axs[1].set_title("MIGs")

    fig.tight_layout()
    return fig


def summarize_preds(preds_df):
    summ = preds_df.groupby(["latent_name", "label_name"]).agg(
        ["mean", "std"]).drop("sample_num", axis="columns")
    print("=== Predictive Performance ===")
    print(summ.to_string())

    fig, axs = plt.subplots(1, 3, figsize=[10, 4])
    i = 0
    for latent_name in sorted(preds_df.latent_name.unique()):
        df = preds_df.loc[preds_df.latent_name == latent_name, :]
        means = df.groupby("label_name").mean().drop(
            "sample_num", axis="columns")
        errs = df.groupby("label_name").std().drop(
            "sample_num", axis="columns")
        means.plot.bar(ax=axs[i], yerr=errs, ylim=(0.2, 1.0), rot=0)
        axs[i].set_title(f"Latent: {latent_name}")
        i += 1
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args()
    if args.test is True:
        print("BIJECTIVE ORACLE")
        print("  random")
        test_bijective_oracle(args.N, args.K, args.n_features)
        print("  predictive")
        test_bijective_oracle(args.N, args.K, args.n_features, predictive=True)
        print()
        print("BIJECTIVE")
        print("  random")
        test_bijective(args.N, args.K)
        print("  predictive")
        test_bijective(args.N, args.K, predictive=True)
        print()
        print()
        print("RANDOM")
        test_random(args.N, args.K, args.n_features)
        print("KINDA PREDICTIVE")
        test_kinda_predictive(args.N, args.K, args.n_features)
        print("PREDICTIVE")
        test_predictive(args.N, args.K, args.n_features)
        print()
    elif args.compute is True:
        compute(args)
    elif args.summarize is True:
        summarize_results(args)
