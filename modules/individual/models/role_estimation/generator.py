import copy

import torch
import torch.nn as nn

from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.emb = nn.Linear(config.d_z, config.d_model * config.seq_len)
        self.pe = PositionalEncoding(config.d_model, config.seq_len)
        self.z = nn.Parameter(torch.randn((1, 1, config.d_model)))

        tre = TransformerEncoder(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.activation,
        )
        self.n_tr = config.n_tr
        self.trs = nn.ModuleList([copy.deepcopy(tre) for _ in range(self.n_tr)])

        self.ff = nn.Linear(config.d_model, 19 * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        B = z.size()[0]
        x = self.emb(z)
        x = x.reshape((B, self._config.seq_len, self._config.d_model))
        x = self.pe(x)
        for i in range(self.n_tr):
            x, attn = self.trs[i](x)

        x = self.sigmoid(self.ff(x))
        x = x.view(B, -1, 19, 2)
        return x, attn