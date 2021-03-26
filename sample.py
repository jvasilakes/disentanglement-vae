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
                all_params.append(latent_params)
                if sent.strip() == '':
                    header = "SAMPLE"
                    zs = [torch.randn(param.z.size())
                          for param in latent_params.values()]
                    z = torch.cat(zs, dim=1).to(vae.device)
                else:
                    header = "RECONSTRUCTION"
                    zs = [param.z for param in latent_params.values()]
                    z = torch.cat(zs, dim=1)
                # Remove <EOS> token
                decoded_tokens = decode(z, vae, idx2word, eos_idx)[:-1]
                all_decoded_tokens.append(decoded_tokens)
                # if "polarity" in latent_params:
                #     polarity_idx = [i for (i, lp) in
                #                     enumerate(latent_params.keys())
                #                     if lp == "polarity"][0]
                #     zs_copy = zs[::]
                #     print(zs_copy is zs)
                #     zs_copy[polarity_idx] *= -1
                #     z_p = torch.cat(zs_copy, dim=1).to(vae.device)
                #     # Remove <EOS> token
                #     decoded_tokens_p = decode(
                #         z_p, vae, idx2word, eos_idx)[:-1]
                #     all_decoded_tokens.append(decoded_tokens_p)

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
                    if param.z.size(1) == 1:
                        zstr = f"{param.z.item():^10.4f}"
                    else:
                        zstr = f"{param.z.norm().item():^10.4f}"
                    zs_strs.append(zstr)
                print(f"|{' '.join(tokens):^{max_len}}|   {' | '.join(zs_strs)} |")  # noqa
            print(''.join(['-'] * (max_len + len(z_names_str) + 7)))
            print()
        except EOFError:
            return


if __name__ == "__main__":
    args = parse_args()
    main(args.params_json)
