import torch
import torch.nn as nn
from modules.individual import IndividualDataTypes
from modules.layers.activation import Activation
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import SpatialTemporalTransformer


class Discriminator(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()

        if data_type == IndividualDataTypes.both:
            self.n_kps = 34  # both
        else:
            self.n_kps = 17  # abs or rel

        self.emb_in_spat = Embedding(self.n_kps * 2, config.d_model)
        self.emb_in_temp = Embedding(config.seq_len, config.d_model)

        self.pe_spat = PositionalEncoding(config.d_model, config.seq_len)
        self.pe_temp = PositionalEncoding(config.d_model, self.n_kps * 2)

        self.sttr = nn.ModuleList()
        self.n_tr = config.n_tr
        for _ in range(config.n_tr):
            self.sttr.append(
                SpatialTemporalTransformer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )

        self.emb_out_spat = Embedding(config.d_model, self.n_kps * 2)
        self.emb_out_temp = Embedding(config.d_model, config.seq_len)
        self.x_norm = nn.LayerNorm(config.seq_len * self.n_kps * 2)

        self.z_layer = nn.Linear(config.d_z, config.seq_len * self.n_kps * 2)

        self.ff = nn.Sequential(
            nn.Linear(config.seq_len * self.n_kps * 2 * 2, config.d_ff),
            Activation(config.activation),
        )
        self.out_layer = nn.Linear(config.d_ff, 1)

    def forward(self, x, z, mask=None):
        B, T, P, D = x.shape  # batch, frame, num_points=17(or 34), dim=2
        x = x.view(B, T, P * D)
        x_spat = x  # spatial(B, T, 34(or 68))
        x_temp = x.permute(0, 2, 1)  # temporal(B, 34(or 68), T)

        mask = mask.view(B, T, P * D)
        mask_spat = mask
        mask_temp = mask.permute(0, 2, 1)

        # embedding
        x_spat = self.emb_in_spat(x_spat)
        x_temp = self.emb_in_temp(x_temp)

        # positional encoding
        x_spat = self.pe_spat(x_spat)
        x_temp = self.pe_temp(x_temp)

        # spatial-temporal transformer
        for i in range(self.n_tr):
            x_spat, x_temp, _, _ = self.sttr[i](x_spat, x_temp, mask_spat, mask_temp)
        x_spat = self.emb_out_spat(x_spat)
        x_temp = self.emb_out_temp(x_temp)
        x = x_spat + x_temp.permute(0, 2, 1)
        x = self.x_norm(x.view(B, -1))

        # z layer
        z = self.z_layer(z)

        # concat x and z
        feature = torch.cat([x, z], dim=1)
        feature = self.ff(feature)

        # last layer
        out = self.out_layer(feature)

        return out, feature
