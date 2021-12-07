import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from hashlib import md5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="path to dSentences.npz")
    parser.add_argument("outdir", type=str, help="where to save result")
    return parser.parse_args()


def main(args):
    dataset = np.load(args.infile, encoding="latin1", allow_pickle=True)
    sents = dataset["sentences_array"]
    labels = dataset["latents_classes"]
    # Don't use dataset["metadata"] to get the latent names,
    #    as they are out of order there.
    latent_names = ["verb_obj_tuple", "obj_sing_pl", "gender",
                    "subj_sing_pl", "sent_type", "nr_person",
                    "pos_neg_verb", "verb_tense", "verb_style"]

    examples = []
    for (sent, labs) in tqdm(zip(sents, labels), total=len(sents)):
        sent = sent.decode("utf-8")
        sent_hash = md5(sent.encode()).hexdigest()
        labs_dict = dict(zip(latent_names, labs))
        labs_dict = {k: int(v) for (k, v) in labs_dict.items()}
        d = {"id": sent_hash, "sentence": sent, **labs_dict}
        examples.append(d)

    # train_split value from 'main_beta_vae.py'
    train_split = 0.75
    metadata = dataset['metadata'][()]
    latents_sizes = metadata['latent_sizes']
    n_syntaxes = np.product(latents_sizes[1:])
    # original dataset processing code splits up the generative factors,
    #    but I split by sentence content
    n_chunks = len(sents) / n_syntaxes
    n_train_chunks = int(n_chunks * train_split)
    n_train_sents = n_train_chunks * n_syntaxes
    n_dev_chunks = int((n_chunks - n_train_chunks) / 2)
    n_dev_sents = n_dev_chunks * n_syntaxes
    trainset = examples[:n_train_sents]
    devset = examples[n_train_sents:(n_train_sents+n_dev_sents)]
    testset = examples[n_train_sents+n_dev_sents:]
    print(f"Train: {len(trainset)}, Dev: {len(devset)}, Test: {len(testset)}")
    print("Total: ", len(trainset) + len(devset) + len(testset))

    print(f"Saving to {args.outdir}")
    zipped = zip([trainset, devset, testset], ["train", "dev", "test"])
    for (dset, setname) in zipped:
        outfile = os.path.join(args.outdir, f"{setname}.jsonl")
        with open(outfile, 'w') as outF:
            for datum in dset:
                json.dump(datum, outF)
                outF.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
