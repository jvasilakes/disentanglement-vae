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
    test_parser.add_argument("-N", type=int, default=100000, required=False,
                             help="""Number of samples to use.""")

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


def compute(args):
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
                                               lab_name, id2labels)
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
        name = re.findall(r'.*_(\w+)_[0-9]+.log', fname)[0]
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
    return clf, precision_recall_fscore_support(V, preds, average="macro")


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
        migs[lab_name] = {"top_2_latents": sorted_names[:2],
                          "MIG": mig_v,
                          "top_2_MIs": sorted_lab_mis[:2],
                          "label_entropy": Hv}
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
    idxs = np.random.randint(0, len(vs), size=int(N//5))
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
    migs_plot, mis_plot = summarize_migs(migs_data)
    plot_outfile = os.path.join(plot_dir, f"MIGs_{args.dataset}.pdf")
    migs_plot.savefig(plot_outfile, dpi=300)
    plot_outfile = os.path.join(plot_dir, f"MIs_{args.dataset}.pdf")
    mis_plot.savefig(plot_outfile, dpi=300)

    preds_data = pd.read_csv(preds_outfile)
    boxplot = summarize_preds(preds_data)
    plot_outfile = os.path.join(plot_dir, f"predictions_{args.dataset}.pdf")
    boxplot.savefig(plot_outfile, dpi=300)


def summarize_migs(migs_data):
    migs = defaultdict(list)
    mis = defaultdict(list)
    for datum in migs_data:
        for latent_name in datum.keys():
            if latent_name == "sample_num":
                continue
            migs[latent_name].append(datum[latent_name]["MIG"])
            mis[latent_name].append(datum[latent_name]["top_2_MIs"][0])

    migs_df = pd.DataFrame(migs)
    print("======== MIGs ========")
    migs_summ_df = migs_df.agg(["mean", "std", "size"]).T
    migs_summ_df.reset_index(inplace=True)
    migs_summ_df.columns = ["latent", "mean", "sd", "N"]
    print(migs_summ_df)
    print()
    # Boxplot of the MIGs
    mig_fig, mig_ax = plt.subplots()
    migs_df.boxplot(ax=mig_ax)
    mig_ax.set_title("MIGs")
    mig_fig.tight_layout()

    mi_df = pd.DataFrame(mis)
    print("======== MIs ========")
    mi_summ_df = mi_df.agg(["mean", "std", "size"]).T
    mi_summ_df.reset_index(inplace=True)
    mi_summ_df.columns = ["latent", "mean", "sd", "N"]
    print(mi_summ_df)
    print()
    # Boxplot of the MIs
    mi_fig, mi_ax = plt.subplots()
    mi_df.boxplot(ax=mi_ax)
    mi_ax.set_title("MIs")
    mi_fig.tight_layout()

    return mig_fig, mi_fig


def summarize_preds(preds_df):
    summ = preds_df.groupby(["latent_name", "label_name"]).agg(
        ["mean", "std"]).drop("sample_num", axis="columns")
    print("=== Predictive Performance ===")
    print(summ)

    fig, ax = plt.subplots()
    preds_df.boxplot(by=["latent_name", "label_name"],
                     column=["precision", "recall", "F1"],
                     rot=90, layout=(1, 3), ax=ax)
    ax.set_title("Precision, Recall, F1")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args()
    if args.test is True:
        print("RANDOM")
        test_random(args.N)
        print("KINDA PREDICTIVE")
        test_kinda_predictive(args.N)
        print("PREDICTIVE")
        test_predictive(args.N)
        print()
        print("BIJECTIVE ORACLE")
        print("  random")
        test_bijective_oracle(args.N)
        print("  predictive")
        test_bijective_oracle(args.N, predictive=True)
        print()
        print("BIJECTIVE")
        print("  random")
        test_bijective(args.N)
        print("  predictive")
        test_bijective(args.N, predictive=True)
        print()
    elif args.compute is True:
        compute(args)
    elif args.summarize is True:
        summarize_results(args)
