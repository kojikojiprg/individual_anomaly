import copy

import torch
import torch.nn as nn
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Decoder as TransformerDecoder
from modules.layers.transformer import Encoder as TransformerEncoder


class Generator(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()
        self.en = Encoder(config)
        self.de = Decoder(config)

    def forward(self, x):
        z, attn_w = self.en(x)
        x = self.de(x, z)
        return x, z, attn_w


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = Embedding(17, config.d_model)
        self.pe = PositionalEncoding(config.d_model, config.seq_len + 1)
        self.z = nn.Parameter(torch.randn((1, 1, config.d_model)))

        tre = TransformerEncoder(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.activation,
        )
        self.n_tr = config.n_tr_e
        self.trs = nn.ModuleList([copy.deepcopy(tre) for _ in range(self.n_tr)])

        self.ff = nn.Linear(config.d_model, config.d_z)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.size()[0]
        x = self.emb(x)
        x = torch.cat([self.z.repeat(B, 1, 1), x], dim=1)
        x = self.pe(x)
        for i in range(self.n_tr):
            x, attn_w = self.trs[i](x)

        z = self.sigmoid(self.ff(x[:, 0]))
        return z, attn_w


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = Embedding(17, config.d_model)
        self.pe = PositionalEncoding(config.d_model, config.seq_len)

        self.linear_z = nn.Linear(config.d_z, config.d_model)

        trd = TransformerDecoder(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
            config.activation,
        )
        self.n_tr = config.n_tr_d
        self.trs = nn.ModuleList([copy.deepcopy(trd) for _ in range(self.n_tr)])

        self.ff = nn.Linear(config.d_model, 17 * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        B = x.size()[0]
        x = self.emb(x)
        x = self.pe(x)

        z = self.linear_z(z)

        for i in range(self.n_tr):
            x, attn_w = self.trs[i](x, z.unsqueeze(1))

        x = self.sigmoid(self.ff(x))
        x = x.view(B, -1, 17, 2)
        return x
