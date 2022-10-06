import torch.nn as nn

from ..layers.embedding import Embedding
from ..layers.positional_encoding import PositionalEncoding
from ..layers.transformer import SpatialTemporalTransformer


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb_spat = Embedding(17 * 2, config.d_model)
        self.emb_temp = Embedding(config.seq_len, config.d_model)

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

        self.fc_spat = Embedding(config.d_model, 17 * 2)
        self.fc_temp = Embedding(config.d_model, config.seq_len)

        self.fc = nn.Linear(config.seq_len * 17 * 2, config.d_z)
        self.norm = nn.LayerNorm(config.d_z)

    def forward(self, x):
        B, T, P, D = x.shape  # batch, frame, num_points=17, dim=2
        x = x.view(B, T, P * D)
        x_spat = x  # spatial(B, T, 34)
        x_temp = x.permute(0, 2, 1)  # temporal(B, 34, T)

        # embedding
        x_spat = self.emb_spat(x_spat)
        x_temp = self.emb_temp(x_temp)

        # positional encoding
        x_spat = self.pe_spat(x_spat)
        x_temp = self.pe_temp(x_temp)

        # spatial-temporal transformer
        for i in range(self.n_sttr):
            x_spat, x_temp, weights_spat, weights_temp = self.sttr[i](x_spat, x_temp)
        x_spat = self.fc_spat(x_spat)
        x_temp = self.fc_temp(x_temp)

        feature = x_spat + x_temp.permute(0, 2, 1)
        z = self.fc(feature.view(B, -1))
        z = self.norm(z)

        return z, feature, weights_spat, weights_temp
