import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from hashlib import md5

import spacy
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="path to dSentences.npz")
    parser.add_argument("outdir", type=str, help="where to save result")
    parser.add_argument("--split_on", type=str,
                        choices=["content", "factors", "random"],
                        help="How to split into train/dev/test.")
    parser.add_argument("--object_tokens_file", type=str, default=None,
                        help="""If specified, get separate annotations
                                for each verb and object, in contrast to
                                the original verb_obj_tuple annotations.""")
    return parser.parse_args()


def main(args):
    dataset = np.load(args.infile, encoding="latin1", allow_pickle=True)
    sents = dataset["sentences_array"]
    labels = dataset["latents_classes"]
    # Don't use dataset["metadata"] to get the latent names,
    #    as they are out of order there.
    latent_names = ["verb_obj_tuple", "obj_sing_pl", "sent_type",
                    "gender", "subj_sing_pl", "nr_person",
                    "pos_neg_verb", "verb_tense", "verb_style"]

    examples = []
    for (sent, labs) in tqdm(zip(sents, labels), total=len(sents)):
        sent = sent.decode("utf-8")
        sent_hash = md5(sent.encode()).hexdigest()
        labs_dict = dict(zip(latent_names, labs))
        labs_dict = {k: int(v) for (k, v) in labs_dict.items()}
        d = {"id": sent_hash, "sentence": sent, **labs_dict}
        examples.append(d)

    if args.object_tokens_file is not None:
        object_tokens = [token.strip() for token
                         in open(args.object_tokens_file)]
        examples = get_verb_object_annotations(examples, object_tokens)

    # train_split value from 'main_beta_vae.py'
    train_split = 0.75
    metadata = dataset['metadata'][()]
    latents_sizes = metadata['latent_sizes']
    n_syntaxes = np.product(latents_sizes[1:])
    if args.split_on == "content":
        n_chunks = len(examples) / n_syntaxes
        n_train_chunks = int(n_chunks * train_split)
        train_idxs = list(range(n_train_chunks * n_syntaxes))
        n_dev_chunks = int((n_chunks - n_train_chunks) / 2)
        dev_start_idx = train_idxs[-1] + 1
        dev_end_idx = dev_start_idx + (n_dev_chunks * n_syntaxes)
        dev_idxs = list(range(dev_start_idx, dev_end_idx))
        test_start_idx = dev_idxs[-1] + 1
        test_idxs = list(range(test_start_idx, len(examples)))

        trainset = [examples[i] for i in train_idxs]
        devset = [examples[i] for i in dev_idxs]
        testset = [examples[i] for i in test_idxs]

    elif args.split_on == "factors":
        raise NotImplementedError("Still working on it...")
        train_chunksize = int(np.floor(n_syntaxes * train_split))
        dev_chunksize = int((n_syntaxes - train_chunksize) / 2)
        test_chunksize = n_syntaxes - train_chunksize - dev_chunksize
        train_idxs = []
        dev_idxs = []
        test_idxs = []
        for i in range(0, len(examples), n_syntaxes):
            train_idxs.extend(list(range(i, i + train_chunksize)))
            dev_idxs.extend(list(range(i + train_chunksize,
                                       i + train_chunksize + dev_chunksize)))
            test_idxs.extend(list(range(
                i + train_chunksize + dev_chunksize,
                i + train_chunksize + dev_chunksize + test_chunksize)))
        trainset = [examples[i] for i in train_idxs]
        devset = [examples[i] for i in dev_idxs]
        testset = [examples[i] for i in test_idxs]

    elif args.split_on == "random":
        trainset, eval_tmp = train_test_split(
            examples, train_size=train_split, shuffle=True, random_state=0)
        devset, testset = train_test_split(
            eval_tmp, train_size=0.5, shuffle=True, random_state=0)

    print(f"Train: {len(trainset)}, Dev: {len(devset)}, Test: {len(testset)}")
    print("Total: ", len(trainset) + len(devset) + len(testset))

    print(f"Saving to {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    zipped = zip([trainset, devset, testset], ["train", "dev", "test"])
    for (dset, setname) in zipped:
        outfile = os.path.join(args.outdir, f"{setname}.jsonl")
        with open(outfile, 'w') as outF:
            for datum in dset:
                json.dump(datum, outF)
                outF.write('\n')


def get_verb_object_annotations(examples, object_tokens):
    """
    The default dSentences annotations lump the verb and object together into
    a single annotation, which makes it impossible to model them separately.
    This function generates separate annoatations for verb and object.

    :param dict examples: full dataset in JSON format
    :param list object_tokens: list of object strings
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Map root form of each object to a unique id
    obj2idx = {}
    lemma2idx = {}
    for obj in object_tokens:
        obj_processed = nlp(obj)[0]
        lemma = obj_processed.lemma_
        try:
            idx = lemma2idx[lemma]
        except KeyError:
            idx = len(lemma2idx)
            lemma2idx[lemma] = idx
        obj2idx[obj] = idx

    new_examples = []
    for ex in examples:
        # There is new verb every 10 indices in the raw data.
        verb_idx = ex["verb_obj_tuple"] // 10
        obj_text = ex["sentence"].split()[-1]
        obj_idx = obj2idx[obj_text]
        ex["verb"] = verb_idx
        ex["object"] = obj_idx
        new_examples.append(ex)

    return new_examples


if __name__ == "__main__":
    args = parse_args()
    main(args)
