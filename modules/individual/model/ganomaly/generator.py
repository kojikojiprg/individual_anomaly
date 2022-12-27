import copy

import torch
import torch.nn as nn

from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Decoder as TransformerDecoder
from modules.layers.transformer import Encoder as TransformerEncoder


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.en = Encoder(config)
        self.de = Decoder(config)

    def forward(self, x, mask):
        z, attn_en = self.en(x, mask)
        x, attn_de = self.de(x, z, mask)
        return x, z, attn_en, attn_de


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = Embedding(config.n_kps, config.d_model)
        self.pe = PositionalEncoding(config.d_model, config.seq_len + 1)
        self.z = nn.Parameter(torch.randn((1, 1, config.d_model)))
        self.z_mask = nn.Parameter(torch.full((1, 1), False), requires_grad=False)

        self.n_heads = config.n_heads
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

    def forward(self, x, mask):
        B = x.size()[0]

        # embedding
        x = self.emb(x)
        x = torch.cat([self.z.repeat(B, 1, 1), x], dim=1)

        # positional encoding
        x = self.pe(x)

        # add mask
        mask = torch.cat([self.z_mask.repeat(B, 1), mask], dim=1)

        # transformer encoder
        for i in range(self.n_tr):
            x, attn_w = self.trs[i](x, key_padding_mask=mask)

        z = self.sigmoid(self.ff(x[:, 0]))
        return z, attn_w


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_kps = config.n_kps

        self.emb = Embedding(config.n_kps, config.d_model)
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

        self.ff = nn.Linear(config.d_model, config.n_kps * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z, mask):
        B = x.size()[0]

        # embedding
        x = self.emb(x)

        # positional encoding
        x = self.pe(x)

        # embedding z
        z = self.linear_z(z)

        # transformer decoder
        for i in range(self.n_tr):
            x, attn_w = self.trs[i](x, z.unsqueeze(1), key_padding_mask=mask)

        x = self.sigmoid(self.ff(x))
        x = x.view(B, -1, self.n_kps, 2)
        return x, attn_w
