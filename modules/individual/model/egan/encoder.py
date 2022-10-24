import torch.nn as nn
from modules.individual import IndividualDataTypes
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import SpatialTemporalTransformer


class Encoder(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()

        if data_type == IndividualDataTypes.both:
            self.n_kps = 34  # both
        else:
            self.n_kps = 17  # abs or rel

        self.emb_spat = Embedding(self.n_kps * 2, config.d_model)
        self.emb_temp = Embedding(config.seq_len, config.d_model)

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

        self.fc_spat = Embedding(config.d_model, self.n_kps * 2)
        self.fc_temp = Embedding(config.d_model, config.seq_len)

        self.fc = nn.Linear(config.seq_len * self.n_kps * 2, config.d_z)
        self.norm = nn.LayerNorm(config.d_z)

    def forward(self, x, mask=None):
        B, T, P, D = x.shape  # batch, frame, num_points=17(or 34), dim=2
        x = x.view(B, T, P * D)
        x_spat = x  # spatial(B, T, 34(or 68))
        x_temp = x.permute(0, 2, 1)  # temporal(B, 34(or 68), T)

        mask = mask.view(B, T, P * D)
        mask_spat = mask
        mask_temp = mask.permute(0, 2, 1)

        # embedding
        x_spat = self.emb_spat(x_spat)
        x_temp = self.emb_temp(x_temp)

        # positional encoding
        x_spat = self.pe_spat(x_spat)
        x_temp = self.pe_temp(x_temp)

        # spatial-temporal transformer
        for i in range(self.n_tr):
            x_spat, x_temp, attn_weights = self.sttr[i](
                x_spat, x_temp, mask_spat, mask_temp
            )

        x_spat = self.fc_spat(x_spat)
        x_temp = self.fc_temp(x_temp)

        x = x_spat + x_temp.permute(0, 2, 1)
        z = self.fc(x.view(B, -1))
        z = self.norm(z)

        return z, attn_weights
