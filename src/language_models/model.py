# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch.utils.data.dataloader
from torch.nn.functional import scaled_dot_product_attention
import numpy as np
import logging
import math
# from src.language_models.utils import PositionalEncoding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.
    ntoken: vocab size
    nip: embedding size
    """

    def __init__(
        self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False
    ):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh",
                                "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
            )
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1, tied=True):
        super().__init__()

        # Input checks
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        assert vocab_size > 0, "vocab_size must be positive"

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tied = tied

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (register as buffer to move with model)
        self.register_buffer(
            'pos_encoding', self._get_pos_encoding(5000, d_model))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=False  # Training script expects (seq, batch, features)
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output layer
        self.decoder = nn.Linear(d_model, vocab_size)

        # Weight tying (if requested)
        if tied:
            self.decoder.weight = self.embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _get_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if not self.tied:
            self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def _causal_mask(self, size, device):
        """Generate causal mask compatible with training script format"""
        return torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)

    def forward(self, input_seq):
        """
        Forward pass for TransformerLM

        Args:
            input_seq: (seq_len, batch_size) - token indices

        Returns:
            output: (seq_len, batch_size, vocab_size) - logits
        """
        seq_len, batch_size = input_seq.shape

        # Input validation
        assert seq_len <= self.pos_encoding.size(
            0), f"Sequence too long: {seq_len} > {self.pos_encoding.size(0)}"

        # Embeddings + positional encoding
        embedded = self.embedding(input_seq) * math.sqrt(self.d_model)
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(1)
        embedded = self.dropout(embedded)

        # Create causal mask
        causal_mask = self._causal_mask(seq_len, input_seq.device)

        # Transformer forward pass
        output = self.transformer(embedded, embedded, tgt_mask=causal_mask)

        # Project to vocabulary
        output = self.decoder(output)

        return output
