# Built-in packages
import random
from collections import namedtuple

# External packages
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel

from model import Discriminator


class VariationalEncoderBert(nn.Module):
    """
    Just send it through BERT and return the pooled output.
    """
    def __init__(self, dropout_rate=0.5, bert_model_str="bert-base-uncased"):
        super(VariationalEncoderBert, self).__init__()
        self._device = torch.device("cpu")
        self.bert_model_str = bert_model_str
        self.dropout_rate = dropout_rate
        self.num_layers = 1
        self.dropout = nn.Dropout(self.dropout_rate)
        self.bert = BertModel.from_pretrained(bert_model_str)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, bert_inputs):
        outputs = self.bert(**bert_inputs)
        return self.dropout(outputs[1])  # pooled output


class VariationalDecoderBert(nn.Module):
    """
    LSTM -> linear -> token_predictions
    """
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, dropout_rate=0.5, emb_matrix=None):
        super(VariationalDecoderBert, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        if num_layers == 1:
            num_layers = 2
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim,
                                 self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, input_ids, lengths, hidden):
        embedded = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        decoded, hidden = self.recurrent(packed, hidden)
        # decoded: [batch_size, max(lengths), self.hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                decoded, batch_first=True)
        # logits: [batch_size, len(inputs), vocab_size]
        logits = self.linear(unpacked)
        return logits, hidden


class VariationalSeq2SeqBert(nn.Module):
    """
    total_latent_dim = 5
    polarity_dsc = Discriminator("polarity", 1, 1)
    modality_dsc = Discriminator("modality", 2, 5)
    # vae will have a 2-dimensional leftover "content" latent space
    vae = VariationalSeq2Seq(encoder, decoder, total_latent_dim,
                             [polarity_dsc, modality_dsc], sos, eos)
    """
    def __init__(self, encoder, decoder, discriminators,
                 latent_dim, bert_model_str,
                 sos_token_idx, eos_token_idx):
        super(VariationalSeq2SeqBert, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_str)

        self._device = torch.device("cpu")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.discriminators = nn.ModuleDict()  # Name: Discriminator
        self.context2params = nn.ModuleDict()  # Name: (mu,logvar) linear layer
        # Total latent dimensions of the discriminators
        self.dsc_latent_dim = 0
        linear_insize = encoder.bert.config.hidden_size
        for dsc in discriminators:
            self.dsc_latent_dim += dsc.latent_dim
            self.discriminators[dsc.name] = dsc
            params_layer = nn.Linear(
                    # 2 for mu, logvar
                    linear_insize, 2 * dsc.latent_dim)
            self.context2params[dsc.name] = params_layer
        assert self.dsc_latent_dim <= self.latent_dim

        # Left over latent dims are treated as a generic "content" space
        if self.dsc_latent_dim < self.latent_dim:
            leftover_latent_dim = self.latent_dim - self.dsc_latent_dim
            leftover_layer = nn.Linear(
                    linear_insize, 2 * leftover_latent_dim)
            self.context2params["content"] = leftover_layer
            assert self.dsc_latent_dim + leftover_latent_dim == self.latent_dim

        self.z2hidden = nn.Linear(
                self.latent_dim, 2 * decoder.hidden_size * decoder.num_layers)
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def trainable_parameters(self):
        return [param for param in self.parameters()
                if param.requires_grad is True]

    def compute_latent_params(self, context):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
            if self.training is True:
                z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            else:
                z = mu
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
            latent_params[name] = Params(z, mu, logvar)
        return latent_params

    def compute_hidden(self, z, batch_size):
        # hidden: [batch_size, 2 * hidden_size * decoder.num_layers]
        # TODO: why am I using tanh? Try with no activation.
        hidden = torch.tanh(self.z2hidden(z))
        # state, cell: [batch_size, hidden_size * decoder.num_layers]
        state, cell = hidden.chunk(2, dim=1)
        # state, cell = [num_layers, batch_size, hidden_size]
        state = state.chunk(self.decoder.num_layers, dim=-1)
        state = torch.stack(state, dim=0)
        cell = cell.chunk(self.decoder.num_layers, dim=-1)
        cell = torch.stack(cell, dim=0)
        return (state, cell)

    def forward(self, bert_inputs, teacher_forcing_prob=0.5):
        # inputs: [batch_size, max(lengths)]
        batch_size = bert_inputs["input_ids"].size(0)
        context = self.encoder(bert_inputs)
        # latent_params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = self.compute_latent_params(context)

        # Forward pass for each discriminator
        dsc_logits = {}
        for (name, dsc) in self.discriminators.items():
            dlogits = dsc(latent_params[name].z)
            dsc_logits[name] = dlogits

        zs = [param.z for param in latent_params.values()]
        z = torch.cat(zs, dim=1)
        decoder_hidden = self.compute_hidden(z, batch_size)
        # decoder_hidden = encoder_hidden

        decoder_input = torch.LongTensor(
                [[self.sos_token_idx]]).to(self.device)
        decoder_input = decoder_input.repeat(batch_size, 1)
        input_lengths = [1] * batch_size
        vocab_size = self.decoder.vocab_size
        target_length = bert_inputs["input_ids"].size(-1)
        # Placeholder for predictions
        out_logits = torch.zeros(
                batch_size, target_length, vocab_size).to(self.device)
        out_logits[:, 0, self.sos_token_idx] = 1.0  # Always output <SOS> first
        out_predictions = torch.zeros(batch_size, target_length, dtype=int)
        out_predictions[:, 0] = self.sos_token_idx
        for i in range(1, target_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            logits = logits.squeeze()
            out_logits[:, i, :] = logits
            use_teacher_forcing = random.random() < teacher_forcing_prob
            if use_teacher_forcing is True:
                target = bert_inputs["input_ids"][:, i]
                decoder_input = torch.unsqueeze(target, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                decoder_input = torch.multinomial(probs, 1)
                # logprobs = torch.log_softmax(logits, dim=-1)
                # decoder_input = logprobs.argmax(-1).unsqueeze(1).detach()
            out_predictions[:, i] = decoder_input.squeeze()

        # decoder_logits: (batch_size, target_length, vocab_size)
        # latent_params: dict({latent_name: Params})
        # dsc_logits: dict({dsc_name: dsc_logits})
        output = {"decoder_logits": out_logits,
                  "latent_params": latent_params,  # Params(z, mu, logvar)
                  "dsc_logits": dsc_logits,
                  "token_predictions": out_predictions}
        return output

    def sample(self, z, max_length=30):
        batch_size = 1
        decoder_hidden = self.compute_hidden(z, batch_size)
        decoder_input = torch.LongTensor([[self.sos_token_idx]]).to(self.device)  # noqa
        input_lengths = [1]
        # Placeholder for predictions
        vocab_size = self.decoder.vocab_size
        decoder_output = torch.zeros(
                1, max_length, vocab_size).to(self.device)
        decoder_output[:, 0, self.sos_token_idx] = 1.0
        for i in range(max_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            logits = logits.squeeze()
            decoder_output[:, i, :] = logits
            probs = torch.softmax(logits, dim=-1)
            decoder_input = torch.multinomial(probs, 1).unsqueeze(0)
        return decoder_output


def build_vae(params, vocab_size, emb_matrix, label_dims, device,
              sos_token_idx, eos_token_idx):
    """
    :param dict params: Dict of parameters stored in config.json
    :param int vocab_size: Number of tokens in the vocabulary
    :param numpy.ndarray emb_matrix: Matrix of embeddings for
                                     each word in vocab. If None,
                                     the model uses random initialization
    :param dict label_dims: Dict of label_names and their dimensionality
    :param torch.device device: Device on which to put the model
    :param int {sos,eos}_token_idx: Index in vocab of <SOS>/<EOS> tokens
    """
    encoder = VariationalEncoderBert(dropout_rate=params["encoder_dropout"],
                                     bert_model_str="bert-base-uncased")
    encoder.set_device(device)

    decoder = VariationalDecoderBert(
            encoder.bert.config.vocab_size, params["embedding_dim"],
            params["hidden_dim"], params["num_rnn_layers"],
            dropout_rate=params["decoder_dropout"],
            emb_matrix=emb_matrix)
    decoder.set_device(device)

    discriminators = []
    for (name, outdim) in label_dims.items():
        if name not in params["latent_dims"]:
            continue
        latent_dim = params["latent_dims"][name]
        dsc = Discriminator(name, latent_dim, outdim)
        dsc.set_device(device)
        discriminators.append(dsc)

    vae = VariationalSeq2SeqBert(encoder, decoder, discriminators,
                                 params["latent_dims"]["total"],
                                 "bert-base-uncased",
                                 sos_token_idx, eos_token_idx)
    vae.set_device(device)
    return vae
