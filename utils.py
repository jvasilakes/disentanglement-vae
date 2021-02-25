import os
import random
import logging
from collections import defaultdict

import torch
import numpy as np
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_params(params):
    valid_params = {
            "name": str,
            "random_seed": int,
            "data_dir": str,
            "lowercase": bool,
            "checkpoint_dir": str,
            "embedding_dim": int,
            "glove_path": str,
            "hidden_dim": int,
            "num_rnn_layers": int,
            "latent_dims": dict,
            "epochs": int,
            "batch_size": int,
            "learn_rate": float,
            "dropout": float,
            "teacher_forcing_prob": float,
            "lambdas": dict,
            "train": bool,
            "validate": bool,
            "test": bool}
    valid = True
    for (key, val) in valid_params.items():
        if key not in params.keys():
            logging.critical(f"parameter file missing '{key}'")
            valid = False
        if not isinstance(params[key], val):
            param_type = type(params[key])
            logging.critical(f"Parameter '{key}' of incorrect type!")
            logging.critical(f"  Expected '{val}' but got '{param_type}'.")
            valid = False
    if valid is False:
        raise ValueError("Found incorrectly specified parameters.")

    for key in params.keys():
        if key not in valid_params.keys():
            logging.warning(f"Ignoring unused parameter '{key}' in parameter file.")  # noqa


def pad_sequence(batch):
    """
    Pad the sequence batch with 0 vectors.
    Compute the original lengths of each sequence.
    Collect labels into a dict of tensors.
    Meant to be used as the value of the `collate_fn`
    argument to `torch.utils.data.DataLoader`.
    """
    seqs = [torch.squeeze(x) for (x, _) in batch]
    seqs_padded = nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)  # 0 = <PAD>
    lengths = torch.LongTensor([len(s) for s in seqs])
    labels = defaultdict(list)
    for (_, y) in batch:
        for label_name in y.keys():
            labels[label_name].append(y[label_name])
    for label_name in labels.keys():
        labels[label_name] = torch.stack(labels[label_name])
    return seqs_padded, labels, lengths


# === RECONSTRUCT AND LOG INPUT ===
def tensor2text(tensor, idx2word, eos_token_idx):
    """
    Given a tensor of word indices, convert it to a list of strings.
    """
    try:
        eos = torch.where(tensor == eos_token_idx)[0][0]
    except IndexError:
        eos = tensor.size(0)
    return [idx2word[i.item()] for i in tensor[:eos+1]]


def get_reconstructions(model, dataset, idx2word, idxs):
    batch = [dataset[i] for i in idxs]
    Xs, _, lengths = pad_sequence(batch)
    Xs = Xs.to(model.device)
    lengths = lengths.to(model.device)
    output = model(Xs, lengths, teacher_forcing_prob=0.0)
    logits = output["decoder_logits"]

    X_text = [' '.join(tensor2text(X, idx2word, model.eos_token_idx))
              for X in Xs.cpu().detach()]
    recon_idxs = logits.argmax(-1)
    recon_text = [' '.join(tensor2text(r, idx2word, model.eos_token_idx))
                  for r in recon_idxs]
    joined = '\n'.join([f"'{x}' ==> '{r}'" for (x, r)
                        in zip(X_text, recon_text)])
    return joined


def log_reconstructions(model, dataset, idx2word, name, epoch, logdir, n=10):
    idxs = np.random.choice(len(dataset),
                            size=n,
                            replace=False)
    # Log inputs and their reconstructions before model training
    recon_file = os.path.join(logdir, f"reconstructions_{name}.log")
    recon_str = get_reconstructions(model, dataset, idx2word, idxs)
    with open(recon_file, 'a') as outF:
        outF.write(f"EPOCH {epoch}\n")
        outF.write(recon_str + '\n')
# ==================================
