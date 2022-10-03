import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .transformer import SpatialTemporalTransformer


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc = nn.Linear(config.d_z, config.seq_len * 17 * 2)

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

    def to(self, device):
        self = super().to(device)
        self.pe_spat.to(device)
        self.pe_temp.to(device)

    def forward(self, z):
        z = self.fc(z)

        B = z.shape[0]
        z = z.view(B, -1, 34)
        z_spat = z  # spatial
        z_temp = z.permute(0, 2, 1)  # temporal

        # embedding
        z_spat = self.emb_spat(z_spat)
        z_temp = self.emb_temp(z_temp)

        # positional encoding
        z_spat = self.pe_spat(z_spat)
        z_temp = self.pe_temp(z_temp)

        # spatial-temporal transformer
        for i in range(self.n_sttr):
            z_spat, z_temp, weights_spat, weights_temp = self.sttr[i](z_spat, z_temp)

        z = torch.matmul(z_spat, z_temp.permute(0, 2, 1))
        z = z.view((B, -1, 17, 2))

        return z, weights_spat, weights_temp
