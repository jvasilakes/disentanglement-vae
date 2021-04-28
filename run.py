# Built-in packages
import os
import csv
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict

# External packages
import torch
import numpy as np
import texar.torch as tx
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score

# Local packages
import utils
import data_utils
import model


torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_json", type=str,
                        help="""Path to JSON file containing experiment
                                parameters.""")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="""If specified, print tqdm progress bars
                                for training and evaluation.""")
    return parser.parse_args()


def reconstruction_loss(targets, logits, target_lengths):
    recon_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=targets, logits=logits, sequence_length=target_lengths)
    return recon_loss


def kl_divergence(mu, logvar):
    kl = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    kl = kl.mean(0).sum()
    return kl


def compute_losses(model, model_outputs, Xbatch, Ybatch, lengths, params,
                   iteration):
    recon_loss = reconstruction_loss(
            Xbatch, model_outputs["decoder_logits"], lengths)

    # Loss and accuracy for each discriminator
    idv_dsc_losses = dict()
    idv_dsc_accs = dict()
    # total loss over all discriminators
    # used in backward pass
    total_dsc_loss = torch.tensor(0.0).to(model.device)
    for (dsc_name, dsc_logits) in model_outputs["dsc_logits"].items():
        dsc = model.discriminators[dsc_name]
        targets = Ybatch[dsc_name].to(model.device)
        dsc_loss = dsc.compute_loss(dsc_logits, targets)
        dsc_acc = dsc.compute_accuracy(dsc_logits, targets)
        idv_dsc_losses[dsc_name] = dsc_loss.item()
        idv_dsc_accs[dsc_name] = dsc_acc.item()
        total_dsc_loss += dsc_loss

    # KL for each latent space
    idv_kls = dict()
    # total kl over all latent spaces
    # used in backward pass
    # total_kl = torch.tensor(0.0).to(model.device)
    total_kl = 0.0  # scalar for logging
    # tensor scalar for backward pass
    total_weighted_kl = torch.tensor(0.0).to(model.device)
    for (latent_name, latent_params) in model_outputs["latent_params"].items():
        kl = kl_divergence(latent_params.mu, latent_params.logvar)
        idv_kls[latent_name] = kl.item()
        # NB we weight the KL term here.
        # This is so we can easily plug in learnable weights later
        try:
            weight = params["lambdas"][latent_name]
        except KeyError:
            weight = params["lambdas"]["default"]

        total_weighted_kl += weight * kl
        total_kl += kl.item()

    # Compute loss function and do backward pass/update parameters
    loss = recon_loss + total_dsc_loss + total_weighted_kl
    output = {"total_loss": loss,  # Scalar tensor
              "recon_loss": recon_loss.item(),  # scalar
              "total_dsc_loss": total_dsc_loss.item(),  # scalar
              "idv_dsc_losses": idv_dsc_losses,  # dict
              "idv_dsc_accs": idv_dsc_accs,  # dict
              "total_kl": total_kl,  # scalar
              "total_weighted_kl": total_weighted_kl.item(),  # scalar
              "idv_kls": idv_kls}  # dict
    return output


def compute_losses_ft(model, model_outputs, Xbatch,
                      lengths, params, iteration):
    """
    Same as compute_losses, but focuses only on the latents and the decoder.
    """
    recon_loss = reconstruction_loss(
            Xbatch, model_outputs["decoder_logits"], lengths)

    # KL for each latent space
    idv_kls = dict()
    # total kl over all latent spaces
    # used in backward pass
    # total_kl = torch.tensor(0.0).to(model.device)
    total_kl = 0.0  # scalar for logging
    # tensor scalar for backward pass
    total_weighted_kl = torch.tensor(0.0).to(model.device)
    for (latent_name, latent_params) in model_outputs["latent_params"].items():
        kl = kl_divergence(latent_params.mu, latent_params.logvar)
        idv_kls[latent_name] = kl.item()
        # NB we weight the KL term here.
        # This is so we can easily plug in learnable weights later
        try:
            weight = params["lambdas"][latent_name]
        except KeyError:
            weight = params["lambdas"]["default"]

        total_weighted_kl += weight * kl
        total_kl += kl.item()

    # Compute loss function and do backward pass/update parameters
    loss = recon_loss + total_weighted_kl
    output = {"total_loss": loss,  # Scalar tensor
              "recon_loss": recon_loss.item(),  # scalar
              "total_kl": total_kl,  # scalar
              "total_weighted_kl": total_weighted_kl.item(),  # scalar
              "idv_kls": idv_kls}  # dict
    return output


def compute_bleu(Xbatch, pred_batch, idx2word, eos_token_idx):
    Xtext = [[utils.tensor2text(X, idx2word, eos_token_idx)[1:-1]]  # RM SOS and EOS   # noqa
             for X in Xbatch.cpu().detach()]
    pred_text = [utils.tensor2text(pred, idx2word, eos_token_idx)[1:-1]
                 for pred in pred_batch.cpu().detach()]
    bleu = bleu_score(pred_text, Xtext)
    return bleu


def log_params(params_dict, example_ids, logdir, dataset_name, epoch):
    """
    :param defaultdict params_dict: {latent_name: {parameter: [p1...pN]}}
    :param str logdir:
    :param str dataset_name:
    :param int epoch:
    """
    metadata_dir = os.path.join(logdir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Log example IDs in the same order as their parameters.
    ids_dir = os.path.join(metadata_dir, "ordered_ids")
    os.makedirs(ids_dir, exist_ok=True)
    ids_outfile = os.path.join(ids_dir, f"{dataset_name}_{epoch}.log")
    with open(ids_outfile, 'w') as outF:
        for i in example_ids:
            outF.write(f"{i}\n")

    for latent_name in params_dict.keys():
        for (param_name, values) in params_dict[latent_name].items():
            param_dir = os.path.join(metadata_dir, param_name)
            os.makedirs(param_dir, exist_ok=True)
            outfile = os.path.join(
                    param_dir, f"{dataset_name}_{latent_name}_{epoch}.log")
            with open(outfile, 'w') as outF:
                writer = csv.writer(outF, delimiter=',')
                for value in values:
                    row = [f"{dim:.4f}" for dim in value]
                    writer.writerow(row)


def trainstep(model, optimizer, dataloader, params, epoch, idx2word,
              verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total discriminator losses over discriminators per step
    total_dsc_losses = []
    # losses, accuracies per discriminator per step
    idv_dsc_losses = defaultdict(list)
    idv_dsc_accs = defaultdict(list)
    # Total KL over latent spaces per step
    total_kls = []
    total_weighted_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)
    idv_ae = defaultdict(list)
    bleus = []
    # Log example IDs in same order as latent parameters
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.train()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    step = epoch * len(dataloader)
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        # output = {"decoder_logits": [batch_size, target_length, vocab_size]
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "token_predictions": [batch_size, target_length]
        output = model(in_Xbatch, lengths,
                       teacher_forcing_prob=params["teacher_forcing_prob"])
        losses_dict = compute_losses(model, output, target_Xbatch,
                                     Ybatch, lengths, params, step)
        total_loss = losses_dict["total_loss"]
        losses.append(total_loss.item())
        recon_losses.append(losses_dict["recon_loss"])
        total_dsc_losses.append(losses_dict["total_dsc_loss"])
        total_kls.append(losses_dict["total_kl"])
        total_weighted_kls.append(losses_dict["total_weighted_kl"])
        for (dsc_name, dsc_loss) in losses_dict["idv_dsc_losses"].items():
            dsc_acc = losses_dict["idv_dsc_accs"][dsc_name]
            idv_dsc_losses[dsc_name].append(dsc_loss)
            idv_dsc_accs[dsc_name].append(dsc_acc)
        for (latent_name, latent_kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(latent_kl)

        # Update model
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        # Measure Autoencoding by reencoding the reconstructed output.
        x_prime = output["token_predictions"].to(model.device)
        output_prime = model(
                x_prime, lengths,
                teacher_forcing_prob=params["teacher_forcing_prob"])

        for (l_name, l_params) in output_prime["latent_params"].items():
            orig_z = output["latent_params"][l_name].z
            z_prime = l_params.z
            diff = torch.norm(z_prime - orig_z, p=None, dim=1).mean()
            idv_ae[l_name].append(diff.item())

        # Measure self-BLEU
        bleu = compute_bleu(target_Xbatch, x_prime, idx2word,
                            model.eos_token_idx)
        bleus.append(bleu)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"EPOCH (ft): {epoch}")
        if step % 5 == 0:
            summary_writer.add_scalar(
                    "total_loss_step", total_loss.item(), step)
            summary_writer.add_scalar(
                    "recon_loss_step", losses_dict["recon_loss"], step)
            for dsc_name in idv_dsc_losses.keys():
                dsc_loss = idv_dsc_losses[dsc_name][-1]
                dsc_acc = idv_dsc_accs[dsc_name][-1]
                summary_writer.add_scalar(
                        f"dsc_loss_step_{dsc_name}", dsc_loss, step)
                summary_writer.add_scalar(
                        f"dsc_acc_step_{dsc_name}", dsc_acc, step)
            for latent_name in idv_kls.keys():
                kl = idv_kls[latent_name][-1]
                summary_writer.add_scalar(
                        f"kl_step_{latent_name}", kl, step)
        step += 1

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar(
            "avg_dsc_loss_all", np.mean(total_dsc_losses), epoch)
    summary_writer.add_scalar("avg_self_bleu", np.mean(bleus), epoch)
    summary_writer.add_scalar(
            "avg_weighted_kl", np.mean(total_weighted_kls), epoch)
    for dsc_name in idv_dsc_losses.keys():
        avg_dsc_loss = np.mean(idv_dsc_losses[dsc_name])
        avg_dsc_acc = np.mean(idv_dsc_accs[dsc_name])
        summary_writer.add_scalar(
                f"avg_dsc_loss_{dsc_name}", avg_dsc_loss, epoch)
        summary_writer.add_scalar(
                f"avg_dsc_acc_{dsc_name}", avg_dsc_acc, epoch)
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)
        avg_ae = np.mean(idv_ae[latent_name])
        summary_writer.add_scalar(
                f"avg_ae_{latent_name}", avg_ae, epoch)

    log_params(all_latent_params, all_sent_ids, logdir, "train", epoch)

    logstr = f"TRAIN ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logstr += f" | DISCRIM: {np.mean(total_dsc_losses):.4f} +/- {np.std(total_dsc_losses):.4f}"  # noqa
    logstr += f" | KL: {np.mean(total_kls):.4f} +/- {np.std(total_kls):.4f}"  # noqa
    logging.info(logstr)

    return model, optimizer


def finetune_trainstep(model, optimizer, dataloader, params, epoch, idx2word,
                       verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs/finetune"

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total KL over latent spaces per step
    total_kls = []
    total_weighted_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)
    idv_ae = defaultdict(list)
    bleus = []
    # Log example IDs in same order as latent parameters
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.train()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    step = epoch * len(dataloader)
    for (i, batch) in enumerate(dataloader):
        # We ignore Ybatch, since we're just finetuning the decoder
        in_Xbatch, target_Xbatch, _, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        # output = {"decoder_logits": [batch_size, target_length, vocab_size]
        #           "latent_params": [Params(z, mu, logvar)] * batch_size
        #           "dsc_logits": {latent_name: [batch_size, n_classes]}
        #           "token_predictions": [batch_size, target_length]
        output = model(in_Xbatch, lengths,
                       teacher_forcing_prob=params["teacher_forcing_prob"])
        losses_dict = compute_losses_ft(model, output, target_Xbatch,
                                        lengths, params, step)
        total_loss = losses_dict["total_loss"]
        losses.append(total_loss.item())
        recon_losses.append(losses_dict["recon_loss"])
        total_kls.append(losses_dict["total_kl"])
        total_weighted_kls.append(losses_dict["total_weighted_kl"])
        for (latent_name, latent_kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(latent_kl)

        # Update model
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        # Measure Autoencoding by reencoding the reconstructed output.
        x_prime = output["token_predictions"].to(model.device)
        output_prime = model(
                x_prime, lengths,
                teacher_forcing_prob=params["teacher_forcing_prob"])

        for (l_name, l_params) in output_prime["latent_params"].items():
            orig_z = output["latent_params"][l_name].z
            z_prime = l_params.z
            diff = torch.norm(z_prime - orig_z, p=None, dim=1).mean()
            idv_ae[l_name].append(diff.item())

        # Measure self-BLEU
        bleu = compute_bleu(target_Xbatch, x_prime, idx2word,
                            model.eos_token_idx)
        bleus.append(bleu)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f"EPOCH (ft): {epoch}")
        if step % 5 == 0:
            summary_writer.add_scalar(
                    "total_loss_step", total_loss.item(), step)
            summary_writer.add_scalar(
                    "recon_loss_step", losses_dict["recon_loss"], step)
            for latent_name in idv_kls.keys():
                kl = idv_kls[latent_name][-1]
                summary_writer.add_scalar(
                        f"kl_step_{latent_name}", kl, step)
        step += 1

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar("avg_self_bleu", np.mean(bleus), epoch)
    summary_writer.add_scalar(
            "avg_weighted_kl", np.mean(total_weighted_kls), epoch)
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)
        avg_ae = np.mean(idv_ae[latent_name])
        summary_writer.add_scalar(
                f"avg_ae_{latent_name}", avg_ae, epoch)

    log_params(all_latent_params, all_sent_ids, logdir, "train", epoch)

    logstr = f"TRAIN ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logstr += f" | KL: {np.mean(total_kls):.4f} +/- {np.std(total_kls):.4f}"  # noqa
    logging.info(logstr)

    return model, optimizer


def evalstep(model, dataloader, params, epoch, idx2word, name="dev",
             verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs"

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total discriminator losses over discriminators per step
    total_dsc_losses = []
    # losses, accuracies per discriminator per step
    idv_dsc_losses = defaultdict(list)
    idv_dsc_accs = defaultdict(list)
    # Total KL over latent spaces per step
    total_kls = []
    total_weighted_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)
    bleus = []
    # Log example IDs and latent params
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.eval()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        in_Xbatch, target_Xbatch, Ybatch, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        # TODO: what should I put for the iteration argument here
        losses_dict = compute_losses(model, output, target_Xbatch,
                                     Ybatch, lengths, params, 100)
        losses.append(losses_dict["total_loss"].item())
        recon_losses.append(losses_dict["recon_loss"])
        total_dsc_losses.append(losses_dict["total_dsc_loss"])
        total_kls.append(losses_dict["total_kl"])
        total_weighted_kls.append(losses_dict["total_weighted_kl"])
        for (dsc_name, dsc_loss) in losses_dict["idv_dsc_losses"].items():
            dsc_acc = losses_dict["idv_dsc_accs"][dsc_name]
            idv_dsc_losses[dsc_name].append(dsc_loss)
            idv_dsc_accs[dsc_name].append(dsc_acc)
        for (latent_name, kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(kl)

        # Measure self-BLEU
        x_prime = output["token_predictions"].to(model.device)
        bleu = compute_bleu(target_Xbatch, x_prime, idx2word,
                            model.eos_token_idx)
        bleus.append(bleu)

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f" ↳ EVAL ({name})")

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar(
            "avg_dsc_loss_all", np.mean(total_dsc_losses), epoch)
    summary_writer.add_scalar("avg_self_bleu", np.mean(bleus), epoch)
    summary_writer.add_scalar(
            "avg_weighted_kl", np.mean(total_weighted_kls), epoch)
    summary_writer.flush()
    for dsc_name in idv_dsc_losses.keys():
        avg_dsc_loss = np.mean(idv_dsc_losses[dsc_name])
        avg_dsc_acc = np.mean(idv_dsc_accs[dsc_name])
        summary_writer.add_scalar(
                f"avg_dsc_loss_{dsc_name}", avg_dsc_loss, epoch)
        summary_writer.add_scalar(
                f"avg_dsc_acc_{dsc_name}", avg_dsc_acc, epoch)
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)

    log_params(all_latent_params, all_sent_ids, logdir, name, epoch)

    logstr = f"{name.upper()} ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logstr += f" | DISCRIM: {np.mean(total_dsc_losses):.4f} +/- {np.std(total_dsc_losses):.4f}"  # noqa
    logging.info(logstr)


def finetune_evalstep(model, dataloader, params, epoch, idx2word, name="dev",
                      verbose=True, summary_writer=None, logdir=None):

    if summary_writer is None:
        summary_writer = SummaryWriter()
    if logdir is None:
        logdir = "logs/finetune"

    # Total loss (recon + discriminator + kl) per step
    losses = []
    # Reconstruction losses per step
    recon_losses = []
    # Total KL over latent spaces per step
    total_kls = []
    total_weighted_kls = []
    # UNWEIGHTED KLs per latent space per step
    idv_kls = defaultdict(list)
    bleus = []
    # Log example IDs and latent params
    all_sent_ids = []
    all_latent_params = defaultdict(lambda: defaultdict(list))

    model.eval()
    if verbose is True:
        pbar = tqdm(total=len(dataloader))
    for (i, batch) in enumerate(dataloader):
        # Ignore Ybatch
        in_Xbatch, target_Xbatch, _, lengths, batch_ids = batch
        in_Xbatch = in_Xbatch.to(model.device)
        target_Xbatch = target_Xbatch.to(model.device)
        lengths = lengths.to(model.device)
        output = model(in_Xbatch, lengths, teacher_forcing_prob=0.0)

        # TODO: what should I put for the iteration argument here
        losses_dict = compute_losses_ft(model, output, target_Xbatch,
                                        lengths, params, 100)
        losses.append(losses_dict["total_loss"].item())
        recon_losses.append(losses_dict["recon_loss"])
        total_kls.append(losses_dict["total_kl"])
        total_weighted_kls.append(losses_dict["total_weighted_kl"])
        for (latent_name, kl) in losses_dict["idv_kls"].items():
            idv_kls[latent_name].append(kl)

        # Measure self-BLEU
        x_prime = output["token_predictions"].to(model.device)
        bleu = compute_bleu(target_Xbatch, x_prime, idx2word,
                            model.eos_token_idx)
        bleus.append(bleu)

        # Log latents
        all_sent_ids.extend(batch_ids)
        for (l_name, l_params) in output["latent_params"].items():
            for (param_name, param_batch) in l_params._asdict().items():
                param_batch = param_batch.detach().cpu().tolist()
                all_latent_params[l_name][param_name].extend(param_batch)

        if verbose is True:
            pbar.update(1)
            pbar.set_description(f" ↳ EVAL ({name})")

    if verbose is True:
        pbar.close()

    summary_writer.add_scalar("avg_loss", np.mean(losses), epoch)
    summary_writer.add_scalar("avg_recon_loss", np.mean(recon_losses), epoch)
    summary_writer.add_scalar("avg_self_bleu", np.mean(bleus), epoch)
    summary_writer.add_scalar(
            "avg_weighted_kl", np.mean(total_weighted_kls), epoch)
    summary_writer.flush()
    for latent_name in idv_kls.keys():
        avg_kl = np.mean(idv_kls[latent_name])
        summary_writer.add_scalar(
                f"avg_kl_{latent_name}", avg_kl, epoch)

    log_params(all_latent_params, all_sent_ids, logdir, name, epoch)

    logstr = f"{name.upper()} ({epoch}) TOTAL: {np.mean(losses):.4f} +/- {np.std(losses):.4f}"  # noqa
    logstr += f" | RECON: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}"  # noqa
    logging.info(logstr)


def run(params_file, verbose=False):
    SOS = "<SOS>"
    EOS = "<EOS>"
    PAD = "<PAD>"
    UNK = "<UNK>"

    params = json.load(open(params_file, 'r'))
    utils.validate_params(params)
    utils.set_seed(params["random_seed"])

    # Set logging directory
    logdir = os.path.join("logs", params["name"])
    finetune = False
    if params["finetune-train"] is True or params["finetune-val"] is True:
        finetune = True
        logdir = os.path.join(logdir, "finetune")
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "run.log")
    print(f"Logging to {logfile}")
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # Log parameters
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"START: {now_str}")
    logging.info("PARAMETERS:")
    for (param, val) in params.items():
        logging.info(f"  {param}: {val}")

    # Set model checkpoint directory
    ckpt_dir = os.path.join(params["checkpoint_dir"], params["name"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    label_keys = [lk for lk in params["latent_dims"].keys() if lk != "total"]
    # Read train data
    train_file = os.path.join(params["data_dir"], "train.jsonl")
    tmp = data_utils.get_sentences_labels(
        train_file, N=params["num_train_examples"], label_keys=label_keys)
    train_sents, train_labs, train_ids, train_lab_counts = tmp
    logging.info("Train label counts:")
    for (labname, values) in train_lab_counts.items():
        logging.info(f"  {labname}: {values}")
    train_sents = data_utils.preprocess_sentences(train_sents, SOS, EOS)
    train_labs, label_encoders = data_utils.preprocess_labels(train_labs)

    # Read validation data
    dev_file = os.path.join(params["data_dir"], "dev.jsonl")
    tmp = data_utils.get_sentences_labels(dev_file, label_keys=label_keys)
    dev_sents, dev_labs, dev_ids, dev_lab_counts = tmp
    dev_sents = data_utils.preprocess_sentences(dev_sents, SOS, EOS)
    # Use the label encoders fit on the train set
    dev_labs, _ = data_utils.preprocess_labels(
            dev_labs, label_encoders=label_encoders)

    vocab_path = os.path.join(logdir, "vocab.txt")
    if finetune is True:
        vocab_path = os.path.join(logdir, "../vocab.txt")
    if params["train"] is True:
        # Get token vocabulary
        vocab = [PAD, UNK] + \
            list(sorted({word for doc in train_sents for word in doc}))
        # Save the vocabulary for this experiment
        with open(vocab_path, 'w') as outF:
            for word in vocab:
                outF.write(f"{word}\n")
    else:
        vocab = [word.strip() for word in open(vocab_path)]
    # word2idx/idx2word are used for encoding/decoding tokens
    word2idx = {word: idx for (idx, word) in enumerate(vocab)}

    if params["reverse_input"] is True:
        noisy_train_sents = data_utils.reverse_sentences(train_sents)
        noisy_dev_sents = data_utils.reverse_sentences(dev_sents)
    else:
        noisy_train_sents = train_sents
        noisy_dev_sents = dev_sents

    # Load glove embeddings, if specified
    # This redefines word2idx/idx2word
    emb_matrix = None
    if params["glove_path"] != "":
        logging.info(f"Loading embeddings from {params['glove_path']}")
        glove, _ = utils.load_glove(params["glove_path"])
        emb_matrix, word2idx = utils.get_embedding_matrix(vocab, glove)
        logging.info(f"Loaded embeddings with size {emb_matrix.shape}")
    idx2word = {idx: word for (word, idx) in word2idx.items()}

    # Always load the train data since we need it to build the model
    train_data = data_utils.DenoisingTextDataset(
            noisy_train_sents, train_sents, train_labs, train_ids,
            word2idx, label_encoders)
    train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=params["batch_size"],
            collate_fn=utils.pad_sequence_denoising)
    logging.info(f"Training examples: {len(train_data)}")
    summary_writer_path = os.path.join("runs", params["name"])
    if finetune is True:
        summary_writer_path = os.path.join(summary_writer_path, "finetune")
    train_writer_path = os.path.join(summary_writer_path, "train")
    train_writer = SummaryWriter(log_dir=train_writer_path)

    if params["validate"] is True or params["finetune-val"] is True:
        dev_data = data_utils.DenoisingTextDataset(
                noisy_dev_sents, dev_sents, dev_labs, dev_ids,
                word2idx, label_encoders)
        dev_dataloader = torch.utils.data.DataLoader(
                dev_data, shuffle=True, batch_size=params["batch_size"],
                collate_fn=utils.pad_sequence_denoising)
        logging.info(f"Validation examples: {len(dev_data)}")
        dev_writer_path = os.path.join(summary_writer_path, "dev")
        dev_writer = SummaryWriter(log_dir=dev_writer_path)

    # Build the VAE
    # label_dims_dict = train_data.y_dims
    # TODO: This isn't right, because it just so happens that latent_dims are
    #   equal to the output dims for negation/uncertainty.
    #   This will fail in general.
    label_dims_dict = {key: val for (key, val) in params["latent_dims"].items()
                       if key != "total"}
    sos_idx = word2idx[SOS]
    eos_idx = word2idx[EOS]
    vae = model.build_vae(params, len(vocab), emb_matrix, label_dims_dict,
                          DEVICE, sos_idx, eos_idx)
    logging.info(vae)

    optimizer = torch.optim.Adam(vae.trainable_parameters(),
                                 lr=params["learn_rate"])

    # If there is a checkpoint at checkpoint_dir, we load it and continue
    # training/evaluating from there.
    # If no checkpoints exist at checkpoint_dir, load_latest_checkpoint
    # will return the same model and opt, and start_epoch=0
    checkpoint_found = False
    logging.info("Trying to load latest model checkpoint from")
    logging.info(f"  {ckpt_dir}")
    vae, optimizer, start_epoch, ckpt_fname = utils.load_latest_checkpoint(
            vae, optimizer, ckpt_dir)
    if ckpt_fname is None:
        logging.warning("No checkpoint found!")
    else:
        checkpoint_found = True
        logging.info(f"Loaded checkpoint '{ckpt_fname}'")
    # If we're finetuning, we want to save the new checkpoints separately
    if finetune is True:
        start_epoch = 0
        ckpt_dir = os.path.join(ckpt_dir, "finetune")
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.info(f"Updating checkpoint directory to '{ckpt_dir}'")

    # Log the experiment parameter file to recreate this run.
    config_logfile = os.path.join(logdir, f"config_epoch{start_epoch}.json")
    with open(config_logfile, 'w') as outF:
        json.dump(params, outF, indent=2)

    if params["train"] is True:
        logging.info("TRAINING")
        logging.info("Ctrl-C to interrupt and save most recent model.")
        if checkpoint_found is False:
            logging.warning("No checkpoint found! Training from base model.")

        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                vae, optimizer = trainstep(
                        vae, optimizer, train_dataloader, params, epoch,
                        idx2word, verbose=verbose, summary_writer=train_writer,
                        logdir=logdir)
                # Log train inputs and their reconstructions
                utils.log_reconstructions(vae, train_data, idx2word,
                                          "train", epoch, logdir, n=20)
                if params["validate"] is True:
                    evalstep(vae, dev_dataloader, params, epoch, idx2word,
                             verbose=verbose, summary_writer=dev_writer,
                             logdir=logdir)
                    # Log dev inputs and their reconstructions
                    utils.log_reconstructions(vae, dev_data, idx2word,
                                              "dev", epoch, logdir, n=20)
            except KeyboardInterrupt:
                break

        # Save the model
        ckpt_fname = f"model_{epoch}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        logging.info(f"Saving trained model to {ckpt_path}")
        torch.save({"model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                   ckpt_path)
        checkpoint_found = True
        start_epoch = epoch

    if params["validate"] is True:
        evalstep(vae, dev_dataloader, params, start_epoch, idx2word,
                 verbose=verbose, summary_writer=dev_writer, logdir=logdir)
        utils.log_reconstructions(vae, dev_data, idx2word,
                                  "dev", start_epoch, logdir, n=30)

    if params["finetune-train"] is True:
        logging.info("Fine-tuning")
        if checkpoint_found is False:
            ValueError("No checkpoint found! Nothing to fine-tune.")
        logging.info("Ctrl-C to interrupt and save most recent model.")

        # Freeze everything but the decoder
        for (name, param) in vae.named_parameters():
            if not name.startswith("decoder"):
                param.requires_grad = False

        epoch_range = range(start_epoch, start_epoch + params["epochs"])
        for epoch in epoch_range:
            try:
                vae, optimizer = finetune_trainstep(
                        vae, optimizer, train_dataloader, params, epoch,
                        idx2word, verbose=verbose, summary_writer=train_writer,
                        logdir=logdir)
                # Log train inputs and their reconstructions
                utils.log_reconstructions(vae, train_data, idx2word,
                                          "train", epoch, logdir, n=20)
                if params["finetune-val"] is True:
                    finetune_evalstep(vae, dev_dataloader, params, epoch,
                                      idx2word, verbose=verbose,
                                      summary_writer=dev_writer,
                                      logdir=logdir)
                    # Log dev inputs and their reconstructions
                    utils.log_reconstructions(vae, dev_data, idx2word,
                                              "dev", epoch, logdir, n=20)
            except KeyboardInterrupt:
                break

        # Save the model
        ckpt_fname = f"model_{epoch}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        logging.info(f"Saving trained model to {ckpt_path}")
        torch.save({"model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch},
                   ckpt_path)
        checkpoint_found = True
        start_epoch = epoch

    if params["finetune-val"] is True:
        finetune_evalstep(vae, dev_dataloader, params, start_epoch, idx2word,
                          verbose=verbose, summary_writer=dev_writer,
                          logdir=logdir)
        utils.log_reconstructions(vae, dev_data, idx2word,
                                  "dev", start_epoch, logdir, n=30)

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"END: {now_str}")


if __name__ == "__main__":
    args = parse_args()
    run(args.params_json, verbose=args.verbose)
