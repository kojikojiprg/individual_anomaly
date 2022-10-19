import torch
import torch.nn as nn
from modules.layers.activation import Activation
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb = Embedding(4, config.d_model)

        self.pe = PositionalEncoding(config.d_model, config.seq_len)

        self.tre = nn.ModuleList()
        self.n_tr = config.n_tr
        for _ in range(self.n_tr):
            self.tre.append(
                Encoder(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )
        self.emb_out = nn.Linear(config.d_model, 4)

        self.z_layer = nn.Linear(config.d_z, config.seq_len * 4)

        self.ff = nn.Sequential(
            nn.Linear(config.seq_len * 4 * 2, config.d_ff),
            Activation(config.activation),
        )
        self.out_layer = nn.Linear(config.d_ff, 1)

    def forward(self, x, z):
        B = x.shape[0]

        # embedding
        x = self.emb(x)

        # positional encoding
        x = self.pe(x)

        # transformer encoder
        for i in range(self.n_tr):
            x, _ = self.tre[i](x)

        x = self.emb_out(x)
        x = x.view(B, -1)

        # z layer
        z = self.z_layer(z)

        # concat x and z
        feature = torch.cat([x, z], dim=1)
        feature = self.ff(feature)

        # last layer
        out = self.out_layer(feature)

        return out, feature
