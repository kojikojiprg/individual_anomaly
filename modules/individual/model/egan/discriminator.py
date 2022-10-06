import torch
import torch.nn as nn

from ..layers.embedding import Embedding
from ..layers.positional_encoding import PositionalEncoding
from ..layers.transformer import SpatialTemporalTransformer


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb_in_spat = Embedding(17 * 2, config.d_model)
        self.emb_in_temp = Embedding(config.seq_len, config.d_model)

        self.pe_spat = PositionalEncoding(config.d_model, config.seq_len)
        self.pe_temp = PositionalEncoding(config.d_model, 17 * 2)

        self.sttr = nn.ModuleList()
        self.n_sttr = config.n_sttr
        for _ in range(config.n_sttr):
            self.sttr.append(
                SpatialTemporalTransformer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )

        self.emb_out_spat = Embedding(config.d_model, 17 * 2)
        self.emb_out_temp = Embedding(config.d_model, config.seq_len)
        self.x_norm = nn.LayerNorm(config.seq_len * 17 * 2)

        self.z_layer = nn.Sequential(
            nn.Linear(config.d_z, config.seq_len * 17 * 2),
            self._get_activation(config.activation),
            nn.LayerNorm(config.seq_len * 17 * 2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_len * 17 * 2 * 2, config.d_ff),
            self._get_activation(config.activation),
            nn.BatchNorm1d(config.d_ff),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.d_ff, config.d_output),
            nn.BatchNorm1d(config.d_output),
        )

    @staticmethod
    def _get_activation(activation):
        if activation == "ReLU":
            return nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            return nn.GELU()
        elif activation == "SELU":
            return nn.SELU(inplace=True)
        else:
            raise NameError

    def forward(self, x, z):
        B, T, P, D = x.shape  # batch, frame, num_points=17, dim=2
        x = x.view(B, T, P * D)
        x_spat = x  # spatial(B, T, 34)
        x_temp = x.permute(0, 2, 1)  # temporal(B, 34, T)

        # embedding
        x_spat = self.emb_in_spat(x_spat)
        x_temp = self.emb_in_temp(x_temp)

        # positional encoding
        x_spat = self.pe_spat(x_spat)
        x_temp = self.pe_temp(x_temp)

        # spatial-temporal transformer
        for i in range(self.n_sttr):
            x_spat, x_temp, weights_spat, weights_temp = self.sttr[i](x_spat, x_temp)
        x_spat = self.emb_out_spat(x_spat)
        x_temp = self.emb_out_temp(x_temp)
        x = x_spat + x_temp.permute(0, 2, 1)
        x = self.x_norm(x.view(B, -1))

        # z layer
        z = self.z_layer(z)

        # concat x and z
        feature = torch.cat([x, z], dim=1)
        feature = self.fc1(feature)

        # last layer
        out = self.fc2(feature)

        return out, feature
