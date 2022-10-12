import torch.nn as nn
from modules.individual import IndividualDataTypes
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Generator(nn.Module):
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
        self.n_tre = config.n_tre
        for _ in range(self.n_tre):
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
        for i in range(self.n_tre):
            z, weights = self.tr[i](z)

        z = self.fc(z)
        z = z.view(B, -1, self.n_kps, 2)  # B, T, 17(or 34), 2

        return z, weights
