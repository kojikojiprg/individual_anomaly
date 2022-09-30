import torch
import torch.nn as nn

from .layers.embedding import Embedding
from .layers.positional_encoding import PositionalEncoding
from .layers.transformer import SpatialTemporalTransformer


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb_spat = Embedding(17 * 2, config.d_emb, config.d_model)
        self.emb_temp = Embedding(config.seq_len, config.d_emb, config.d_model)

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

        self.fc = nn.Linear(config.seq_len * 17 * 2, config.d_output)

    def to(self, device):
        self = super().to(device)
        self.pe_spat.to(device)
        self.pe_temp.to(device)

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

        feature = torch.matmul(x_spat, x_temp.permute(0, 2, 1))
        x = self.fc(feature.view(B, -1))

        return x, feature, weights_spat, weights_temp
