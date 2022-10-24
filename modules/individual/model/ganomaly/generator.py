import torch.nn as nn
from modules.individual import IndividualDataTypes
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder
from modules.layers.transformer import SpatialTemporalTransformer


class Generator(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()
        self.en = Encoder(config, data_type)
        self.de = Decoder(config, data_type)

    def forward(self, x):
        z, attn = self.en(x)
        x = self.de(z)

        return x, z, attn


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
        self.n_tr = config.n_tr_e
        for _ in range(self.n_tr):
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
        for i in range(self.n_tr):
            x_spat, x_temp, attn_weights = self.sttr[i](x_spat, x_temp)
        x_spat = self.fc_spat(x_spat)
        x_temp = self.fc_temp(x_temp)

        x = x_spat + x_temp.permute(0, 2, 1)
        x = self.fc(x.view(B, -1))
        z = self.norm(x)

        return z, attn_weights


class Decoder(nn.Module):
    def __init__(self, config, data_type):
        super().__init__()

        if data_type == IndividualDataTypes.both:
            self.n_kps = 34  # both
        else:
            self.n_kps = 17  # abs or rel
        self.seq_len = config.seq_len
        self.d_model = config.d_model

        self.emb = Embedding(config.d_z, config.seq_len * config.d_model)

        self.pe = PositionalEncoding(config.d_model, config.seq_len)

        self.tr = nn.ModuleList()
        self.n_tr = config.n_tr_d
        for _ in range(self.n_tr):
            self.tr.append(
                TransformerEncoder(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )

        self.fc = nn.Linear(config.d_model, self.n_kps * 2)

    def forward(self, z):
        B = z.shape[0]

        # embedding
        z = self.emb(z)
        z = z.view(B, self.seq_len, self.d_model)  # B, T, D

        # positional encoding
        z = self.pe(z)

        # spatial-temporal transformer
        for i in range(self.n_tr):
            z, _ = self.tr[i](z)

        z = self.fc(z)
        x = z.view(B, -1, self.n_kps, 2)  # B, T, 17, 2

        return x
