# Built-in packages
import random
from collections import namedtuple

# External packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    """
    embedding -> dropout -> LSTM
    """
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers,
                 dropout_rate=0.5, emb_matrix=None):
        super(VariationalEncoder, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        if emb_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(emb_matrix))
            self.embedding.weight.requires_grad = False
            self.vocab_size, self.emb_dim = emb_matrix.shape
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.recurrent = nn.LSTM(self.emb_dim, self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout=self.dropout_rate, batch_first=True)

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, inputs, lengths, hidden):
        # inputs: [batch_size, max(lengths)]
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, max(lengths), self.emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False)
        # packed: [sum(lengths), self.emb_dim]
        encoded, hidden = self.recurrent(packed, hidden)
        # encoded: [batch_size, max(lengths), self.hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        unpacked, lengths_unpacked = nn.utils.rnn.pad_packed_sequence(
                encoded, batch_first=True)
        # unpacked: [batch_size, max(lengths), hidden_size]
        return unpacked, hidden

    def init_hidden(self, batch_size):
        # Initialize the LSTM state.
        # One for hidden and one for the cell
        return (torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=self.device),
                torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=self.device))


class VariationalDecoder(nn.Module):
    """
    LSTM -> linear -> token_predictions
    """
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, dropout_rate=0.5, emb_matrix=None):
        super(VariationalDecoder, self).__init__()
        self._device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
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

    def forward(self, inputs, lengths, hidden):
        embedded = self.dropout(self.embedding(inputs))
        # embedded: [batch_size, len(inputs), emb_dim]
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


class Discriminator(nn.Module):
    def __init__(self, name, latent_dim, output_dim):
        super(Discriminator, self).__init__()
        self._device = torch.device("cpu")
        self.name = name
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(latent_dim, output_dim)
        assert self.output_dim > 0
        if self.output_dim == 1:
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.activation = torch.sigmoid
            self.activation_args = []
        else:
            self.loss_fn = F.cross_entropy
            self.activation = torch.softmax
            self.activation_args = [-1]

    @property
    def device(self):
        return self._device

    def set_device(self, value):
        assert isinstance(value, torch.device)
        self._device = value
        self.to(value)

    def forward(self, inputs):
        return self.linear(inputs)

    def compute_loss(self, logits, targets):
        if self.output_dim > 1:
            targets = targets.squeeze()
        return self.loss_fn(logits, targets)

    def compute_accuracy(self, logits, targets):
        probs = self.activation(logits, *self.activation_args)
        if probs.size(1) == 1:
            preds = (probs > 0.5).long().squeeze()
        else:
            preds = probs.argmax(-1).squeeze()
        targets = targets.squeeze()
        acc = torch.mean((preds == targets).float())
        return acc


class VariationalSeq2Seq(nn.Module):
    """
    total_latent_dim = 5
    polarity_dsc = Discriminator("polarity", 1, 1)
    modality_dsc = Discriminator("modality", 2, 5)
    # vae will have a 2-dimensional leftover "content" latent space
    vae = VariationalSeq2Seq(encoder, decoder, total_latent_dim,
                             [polarity_dsc, modality_dsc], sos, eos)
    """
    def __init__(self, encoder, decoder, latent_dim, discriminators,
                 sos_token_idx, eos_token_idx):
        super(VariationalSeq2Seq, self).__init__()
        self._device = torch.device("cpu")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.discriminators = nn.ModuleDict()  # Name: Discriminator
        self.context2params = nn.ModuleDict()  # Name: (mu,logvar) linear layer
        # Total latent dimensions of the discriminators
        self.dsc_latent_dim = 0
        for dsc in discriminators:
            self.dsc_latent_dim += dsc.latent_dim
            self.discriminators[dsc.name] = dsc
            params_layer = nn.Linear(
                    encoder.hidden_size * encoder.num_layers,
                    # TEST encoder.hidden_size,
                    2 * dsc.latent_dim)
            self.context2params[dsc.name] = params_layer
        assert self.dsc_latent_dim <= self.latent_dim

        # Left over latent dims are treated as a generic "content" space
        if self.dsc_latent_dim < self.latent_dim:
            leftover_latent_dim = self.latent_dim - self.dsc_latent_dim
            leftover_layer = nn.Linear(
                    encoder.hidden_size * encoder.num_layers,
                    # TEST encoder.hidden_size,
                    2 * leftover_latent_dim)
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

    def encode(self, inputs, lengths):
        batch_size = inputs.size(0)
        # state, cell: [num_layers, batch_size, hidden_size]
        state, cell = self.encoder.init_hidden(batch_size)
        # encoded: [batch_size, max(lengths), hidden_size]
        encoded, (hidden_state, hidden_cell) = self.encoder(
                inputs, lengths, (state, cell))
        # context: [batch_size, num_layers * hidden_size]
        # TEST state_context = hidden_state.view(batch_size, -1)
        # TEST state_context = hidden_state[-1, :, :]  # last layer
        state_context = torch.cat([layer for layer in hidden_state], dim=1)
        return encoded, state_context, (hidden_state, hidden_cell)

    def compute_latent_params(self, context):
        latent_params = dict()
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        for (name, layer) in self.context2params.items():
            params = layer(context)
            mu, logvar = params.chunk(2, dim=1)
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
        # TEST state = state.reshape(self.decoder.num_layers, batch_size, -1)
        state = state.chunk(self.decoder.num_layers, dim=-1)
        state = torch.stack(state, dim=0)
        # TEST cell = cell.reshape(self.decoder.num_layers, batch_size, -1)
        cell = cell.chunk(self.decoder.num_layers, dim=-1)
        cell = torch.stack(cell, dim=0)
        return (state, cell)

    def forward(self, inputs, lengths, teacher_forcing_prob=0.5):
        # inputs: [batch_size, max(lengths)]
        batch_size = inputs.size(0)
        encoded, context, encoder_hidden = self.encode(inputs, lengths)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
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
        target_length = inputs.size(-1)
        # Placeholder for predictions
        out_logits = torch.zeros(
                batch_size, target_length, vocab_size).to(self.device)
        out_logits[:, 0, self.sos_token_idx] = 1.0  # Always output <SOS> first
        for i in range(1, target_length):
            # logits: [batch_size, 1, vocab_size]
            logits, decoder_hidden = self.decoder(
                    decoder_input, input_lengths, decoder_hidden)
            out_logits[:, i, :] = logits.squeeze()
            use_teacher_forcing = random.random() < teacher_forcing_prob
            if use_teacher_forcing is True:
                # target = inputs[:, i-1:i]
                target = inputs[:, i]
                decoder_input = torch.unsqueeze(target, 1)
            else:
                logprobs = torch.log_softmax(out_logits, dim=-1)
                decoder_input = logprobs.argmax(-1).detach()
        # decoder_logits: (batch_size, target_length, vocab_size)
        # latent_params: dict({latent_name: Params})
        # dsc_logits: dict({dsc_name: dsc_logits})
        output = {"decoder_logits": out_logits,
                  "latent_params": latent_params,  # Params(z, mu, logvar)
                  "dsc_logits": dsc_logits}
        return output

    def sample(self, z, max_length=30):
        raise NotImplementedError
