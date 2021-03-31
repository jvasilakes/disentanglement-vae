import os
import json
import logging
import argparse

# External packages
import torch

# Local imports
import utils
import data_utils
import model


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


def encode(sentence, vae, SOS, EOS, lowercase, doc2tensor_fn, word2idx):
    preprocessed = data_utils.preprocess_sentences(
            [sentence], SOS, EOS, lowercase=lowercase)[0]
    tensorized = doc2tensor_fn(preprocessed, word2idx).to(vae.device)
    tensorized = tensorized.squeeze(0)

    length = torch.tensor([len(preprocessed)]).to(vae.device)
    encoded, context, _ = vae.encode(tensorized, length)
    return context


def decode(z, vae, idx2word, eos_idx):
    out_logits = vae.sample(z)
    pred_idxs = out_logits.argmax(-1).squeeze()
    tokens = utils.tensor2text(pred_idxs, idx2word, eos_idx)
    return tokens


def interpolate(vae, context1, context2, latent_name):
    params1 = vae.compute_latent_params(context1)
    params2 = vae.compute_latent_params(context2)
    T = 5
    out_params = []
    for i in range(T):
        t = i / T
        zt = (params1[latent_name].z * (1 - t)) + (params2[latent_name].z * t)
        print(params1[latent_name].z)
        print(params2[latent_name].z)
        print(t)
        print(zt)
        print()
        params_copy = params1[latent_name]._asdict()
        params_copy['z'] = zt
        new_params = params1[latent_name].__class__(**params_copy)
        all_params_copy = dict(params1)
        all_params_copy[latent_name] = new_params
        out_params.append(all_params_copy)
    out_params.append(params2)
    return out_params


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
    train_sents, train_labs, train_lab_counts = data_utils.get_sentences_labels(  # noqa
        train_file, N=params["num_train_examples"])
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    vocab_file = os.path.join(logdir, "vocab.txt")
    vocab = [word.strip() for word in open(vocab_file)]
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}
    train_data = data_utils.LabeledTextDataset(
            train_sents, train_labs, word2idx, label_encoders)

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
            vae, optimizer, ckpt_dir)
    logging.info(f"Loaded checkpoint from '{ckpt_fname}'")

    logging.info(f"Successfully loaded model {params['name']}")
    logging.info(vae)
    vae.eval()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Enter a sentence at the prompt.")
    print("Ctrl-D to quit.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    while True:
        try:
            sent = input("> ")
            context = encode(sent, vae, SOS, EOS, params["lowercase"],
                             train_data.doc2tensor, word2idx)
            all_decoded_tokens = []
            all_params = []
            for i in range(5):
                latent_params = vae.compute_latent_params(context)
                if sent.strip() == '':
                    header = "SAMPLE"
                    params_copy = dict(latent_params)
                    for (name, param) in params_copy.items():
                        param_dict = param._asdict()
                        param_dict['z'] = torch.randn(param.z.size()).to(vae.device)  # noqa
                        params_copy[name] = param.__class__(**param_dict)
                    latent_params = params_copy
                    all_params.append(latent_params)
                    zs = [param.z for param in latent_params.values()]
                    z = torch.cat(zs, dim=1)
                    # Remove <EOS> token
                    decoded_tokens = decode(z, vae, idx2word, eos_idx)[:-1]
                    all_decoded_tokens.append(decoded_tokens)
                elif " || " in sent:
                    header = "INTERPOLATE"
                    sent1, sent2 = sent.split("||")
                    context1 = encode(sent1, vae, SOS, EOS,
                                      params["lowercase"],
                                      train_data.doc2tensor, word2idx)
                    context2 = encode(sent2, vae, SOS, EOS,
                                      params["lowercase"],
                                      train_data.doc2tensor, word2idx)
                    all_latent_params = interpolate(
                        vae, context1, context2, "polarity")
                    for lps in all_latent_params:
                        all_params.append(lps)
                        zs = [param.z for param in latent_params.values()]
                        z = torch.cat(zs, dim=1)
                        # Remove <EOS> token
                        decoded_tokens = decode(z, vae, idx2word, eos_idx)[:-1]
                        all_decoded_tokens.append(decoded_tokens)
                    break
                else:
                    header = "RECONSTRUCTION"
                    all_params.append(latent_params)
                    zs = [param.z for param in latent_params.values()]
                    z = torch.cat(zs, dim=1)
                    # Remove <EOS> token
                    decoded_tokens = decode(z, vae, idx2word, eos_idx)[:-1]
                    all_decoded_tokens.append(decoded_tokens)

            max_len = max([len(' '.join(tokens))
                           for tokens in all_decoded_tokens])
            max_len += 2  # For SOS and EOS tokens.
            z_names = [f"{name:^10}" for name in latent_params.keys()]
            z_names_str = ' | '.join(z_names)
            print(f"|{header:^{max_len}}|   {z_names_str} |")
            print(''.join(['-'] * (max_len + len(z_names_str) + 7)))
            for (tokens, l_params) in zip(all_decoded_tokens, all_params):
                zs_strs = []
                for (name, param) in l_params.items():
                    try:
                        dsc = vae.discriminators[name]
                        logits = dsc.forward(param.z)
                        pred = dsc.predict(logits).item()
                    except KeyError:
                        pred = "_"
                    if param.z.size(1) == 1:
                        zstr = f"{param.z.item():^6.4f} ({pred})"
                    else:
                        zstr = f"{param.z.norm().item():^6.4f} ({pred})"
                    zs_strs.append(zstr)
                print(f"|{' '.join(tokens):^{max_len}}|   {' | '.join(zs_strs)} |")  # noqa
            print(''.join(['-'] * (max_len + len(z_names_str) + 7)))
            print()
        except EOFError:
            return


if __name__ == "__main__":
    args = parse_args()
    main(args.params_json)
