import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs='+', type=str, required=True,
                        help="""Two or more datasets to combine. Specify the
                                directories containing {train,dev,test}.jsonl.
                        """)
    parser.add_argument("--dataset_names", nargs='+', type=str, required=True,
                        help="""Names of datasets specified to --data_dirs.""")
    parser.add_argument("--Ns", nargs='+', type=int, required=True,
                        help="""Number of examples to load per dataset.
                                -1 for all.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the combined datasets.""")
    return parser.parse_args()


def main(args):
    if len(args.data_dirs) < 2:
        raise ValueError("Must specify more than 1 data_dirs.")
    if len(args.data_dirs) != len(args.dataset_names):
        raise ValueError("Lengths of data_dirs and dataset_names don't match.")
    os.makedirs(args.outdir, exist_ok=False)

    setnames = ["train", "dev", "test"]
    filepaths = dict(zip(setnames, [[] for i in range(len(setnames))]))
    for setname in setnames:
        for datadir in args.data_dirs:
            fname = f"{setname}.jsonl"
            filepath = os.path.join(datadir, fname)
            if not os.path.exists(filepath):
                raise OSError(f"Expected file at '{filepath}'")
            filepaths[setname].append(filepath)

    for setname in setnames:
        print(f"Merging {setname}")
        # Generator over the merged lines
        if setname == "train":
            Ns = args.Ns
        else:
            Ns = [-1 for _ in range(len(filepaths[setname]))]
        print(f"  {setname} Ns: {Ns}")
        merged = merge_datasets(filepaths[setname],
                                dataset_names=args.dataset_names,
                                Ns=args.Ns)
        outpath = os.path.join(args.outdir, f"{setname}.jsonl")
        with open(outpath, 'w') as outF:
            for datum in merged:
                json.dump(datum, outF)
                outF.write('\n')


def merge_datasets(filepaths, dataset_names=[], Ns=[]):
    assert len(filepaths) == len(dataset_names) == len(Ns)
    to_merge = []
    keys_per_dataset = dict(zip(filepaths,
                                [set() for _ in range(len(filepaths))]))
    seen_ids = set()
    for (fpath, name, N) in zip(filepaths, dataset_names, Ns):
        tmp = []
        for line in open(fpath, 'r'):
            datum = json.loads(line)
            if datum["id"] in seen_ids:
                # duplicate sentence
                continue
            seen_ids.add(datum["id"])
            datum["source_dataset"] = name
            keys_per_dataset[fpath].update(set(datum.keys()))
            tmp.append(datum)
        to_merge.extend(tmp[:N])

    keep_keys = set.intersection(*keys_per_dataset.values())
    assert "sentence" in keep_keys

    for datum in to_merge:
        yield dict((k, datum[k]) for k in keep_keys)


if __name__ == "__main__":
    args = parse_args()
    main(args)
