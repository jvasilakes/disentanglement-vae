import os
import json
import shlex
import logging
import argparse

# External packages
import torch
import torch.distributions as D

# Local imports
from vae import utils, data_utils, model


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Path to JSON file containing
                                experiment parameters.""")
    return parser.parse_args()


def encode(sentence, vae, SOS, EOS, lowercase, doc2tensor_fn):
    preprocessed = data_utils.preprocess_sentences(
            [sentence], SOS, EOS, lowercase=lowercase)[0]
    tensorized = doc2tensor_fn(preprocessed).to(vae.device)
    tensorized = tensorized.squeeze(0)

    length = torch.tensor([len(preprocessed)]).to(vae.device)
    encoded, context, _ = vae.encode(tensorized, length)
    return context


def decode(z, vae, idx2word, eos_idx):
    out_logits = vae.sample(z)
    pred_idxs = out_logits.argmax(-1).squeeze()
    tokens = utils.tensor2text(pred_idxs, idx2word, eos_idx)
    return tokens


def compute_interpolatation(vae, context1, context2, latent_name):
    params1 = vae.compute_latent_params(context1)
    params2 = vae.compute_latent_params(context2)
    T = 5
    out_params = []
    for i in range(T):
        t = i / T
        zt = (params1[latent_name].z * (1 - t)) + (params2[latent_name].z * t)
        params_copy = params1[latent_name]._asdict()
        params_copy['z'] = zt
        new_params = params1[latent_name].__class__(**params_copy)
        all_params_copy = dict(params1)
        all_params_copy[latent_name] = new_params
        out_params.append(all_params_copy)
    out_params.append(params2)
    return out_params


def parse_input(input_args):
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.set_defaults(cmd="")
    subparsers = cmd_parser.add_subparsers(
        help="reconstruct, sample, or interpolate")

    rec_parser = subparsers.add_parser(
        "reconstruct", description="Reconstruct the given sentence.")
    rec_parser.set_defaults(cmd="reconstruct")
    rec_parser.add_argument("sentence", type=str,
                            help="sentence to reconstruct")
    rec_parser.add_argument("--polz", type=float, required=False,
                            help="scalar z value for polarity latent")
    rec_parser.add_argument("--uncz", type=float, required=False,
                            help="scalar z value for uncertainty latent")
    rec_parser.add_argument("-n", type=int, required=False, default=1,
                            help="Number of reconstructions to output.")

    samp_parser = subparsers.add_parser(
        "sample", description="Sample n times from the VAE.")
    samp_parser.set_defaults(cmd="sample")
    samp_parser.add_argument("n", type=int,
                             help="Number of sentences to sample.")
    samp_parser.add_argument("--polz", type=float, required=False,
                             help="scalar z value for polarity latent")
    samp_parser.add_argument("--uncz", type=float, required=False,
                             help="scalar z value for uncertainty latent")

    int_parser = subparsers.add_parser(
        "interpolate",
        description="""Compute n interpolatated latents between
                       between the given sentences and decode them.""")
    int_parser.set_defaults(cmd="interpolate")
    int_parser.add_argument("sentence1", type=str)
    int_parser.add_argument("sentence2", type=str)

    args = cmd_parser.parse_args(input_args)
    return args


def reconstruct(sentence, polz, uncz, vae, do_lowercase,
                doc2tensor_fn, idx2word, n=5):
    context = encode(sentence, vae, "<SOS>", "<EOS>",
                     do_lowercase, doc2tensor_fn)
    all_zs = []
    all_decoded_tokens = []
    for i in range(n):
        latent_params = vae.compute_latent_params(context)
        d = {name: param.z for (name, param) in latent_params.items()}
        if polz is not None:
            d["polarity"] = torch.tensor([[polz]]).to(vae.device)
        if uncz is not None:
            d["uncertainty"] = torch.tensor([[uncz]]).to(vae.device)
        all_zs.append(d)
        zs = [param.z for param in latent_params.values()]
        z = torch.cat(zs, dim=1)
        decoded_tokens = decode(z, vae, idx2word, vae.eos_token_idx)[:-1]
        all_decoded_tokens.append(decoded_tokens)
    return all_decoded_tokens, all_zs


def sample(n, polz, uncz, vae, idx2word):

    latent_names_sizes = {name: layer.out_features // 2
                          for (name, layer) in vae.context2params.items()}
    all_zs = []
    all_decoded_tokens = []
    for i in range(n):
        zs_dict = {}
        for (name, dim) in latent_names_sizes.items():
            mu = torch.zeros(dim)
            sigma = torch.eye(dim)
            d = D.MultivariateNormal(mu, sigma)
            if name == "polarity":
                if polz is None:
                    z = d.sample().unsqueeze(0).to(vae.device)
                else:
                    z = torch.tensor([[polz]]).to(vae.device)
            elif name == "uncertainty":
                if uncz is None:
                    z = d.sample().unsqueeze(0).to(vae.device)
                else:
                    z = torch.tensor([[uncz]]).to(vae.device)
            elif name == "content":
                z = d.sample().unsqueeze(0).to(vae.device)
            zs_dict[name] = z
        all_zs.append(zs_dict)
        z_values = [torch.tensor(z) for z in zs_dict.values()]
        z = torch.cat(z_values, dim=1)
        # Remove <EOS> token
        decoded_tokens = decode(z, vae, idx2word, vae.eos_token_idx)[:-1]
        all_decoded_tokens.append(decoded_tokens)
    return all_decoded_tokens, all_zs


def interpolate(sentence1, sentence2, vae):
    raise NotImplementedError()


def print_decoded_tokens(vae, decoded_tokens, all_zs, header):
    max_len = max([len(' '.join(tokens))
                   for tokens in decoded_tokens])
    max_len += 2  # For SOS and EOS tokens.
    z_names = [f"{name:^12}" for name in all_zs[0].keys()]
    z_names_str = ' | '.join(z_names)
    print(f"|{header:^{max_len}}|   {z_names_str} |")
    print(''.join(['-'] * (max_len + len(z_names_str) + 7)))
    for (tokens, zs) in zip(decoded_tokens, all_zs):
        zs_strs = []
        for (name, z) in zs.items():
            try:
                dsc = vae.discriminators[name]
                logits = dsc.forward(z)
                pred = dsc.predict(logits).item()
            except KeyError:
                pred = "_"
            if z.size(1) == 1:
                zstr = f"{z.item():^8.4f} ({pred})"
            else:
                zstr = f"{z.norm().item():^8.4f} ({pred})"
            zs_strs.append(zstr)
        print(f"|{' '.join(tokens):^{max_len}}|   {' | '.join(zs_strs)} |")  # noqa
    print(''.join(['-'] * (max_len + len(z_names_str) + 7)))
    print()


def main(params_file):
    logging.basicConfig(level=logging.INFO)

    SOS = "<SOS>"
    EOS = "<EOS>"

    params = json.load(open(params_file, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])

    logdir = os.path.join("logs", params["name"])

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No model found at {params['checkpoint_dir']}")

    # Load the train data so we can fully specify the model
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    tmp = data_utils.get_sentences_labels(
        train_file, N=params["num_train_examples"])
    train_sents, train_labs, train_ids, train_lab_counts = tmp
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    vocab_file = os.path.join(logdir, "vocab.txt")
    vocab = [word.strip() for word in open(vocab_file)]
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    train_data = data_utils.DenoisingTextDataset(
        train_sents, train_sents, train_labs, train_ids,
        word2idx, label_encoders)

    # Get word embeddings if specified
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Build the VAE
    label_dims_dict = train_data.y_dims
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    vae = model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    optimizer = torch.optim.Adam(
            vae.trainable_parameters(), lr=params["learn_rate"])

    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        raise OSError(f"No checkpoint found at '{ckpt_dir}'!")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir, map_location=DEVICE)
    logging.info(f"Loaded checkpoint from '{ckpt_fname}'")

    logging.info(f"Successfully loaded model {params['name']}")
    logging.info(vae)
    vae.eval()

    cmd_dict = {"reconstruct": (reconstruct,
                                {"do_lowercase": params["lowercase"],
                                 "doc2tensor_fn": train_data.doc2tensor}),
                "sample": (sample, {}),
                "interpolate": (interpolate, )}

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Enter a sentence at the prompt.")
    print("Ctrl-D to quit.")
    print()
    print("Help")
    print("  reconstruct -h")
    print("  sample -h")
    print("  interpolate -h")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()

    while True:
        try:

            inp = input("> ")
            inp = shlex.split(inp)
            parsed = parse_input(inp)

            value = cmd_dict.get(parsed.cmd)
            if value is None:
                print(f"Unknown command {parsed.cmd}...")
                continue
            action, cmd_kwargs = value
            header = parsed.cmd.upper()

            kwargs = vars(parsed)
            kwargs.pop("cmd")
            kwargs.update(cmd_kwargs)
            decoded_tokens, zs = action(**kwargs, vae=vae, idx2word=idx2word)
            print_decoded_tokens(vae, decoded_tokens, zs, header)

        except EOFError:
            return

        except SystemExit:
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args.params_json)
