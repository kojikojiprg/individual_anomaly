import torch.nn as nn
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.emb = Embedding(4, config.d_model)

        self.pe = PositionalEncoding(config.d_model, config.seq_len)

        self.tre = nn.ModuleList()
        self.n_tre = config.n_tr
        for _ in range(self.n_tre):
            self.tre.append(
                TransformerEncoder(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )

        self.fc = nn.Linear(config.seq_len * config.d_model, config.d_z)
        self.norm = nn.LayerNorm(config.d_z)

    def forward(self, x):
        B = x.shape[0]

        # embedding
        x = self.emb(x)

        # positional encoding
        x = self.pe(x)

        # spatial-temporal transformer
        for i in range(self.n_tre):
            x, weights = self.tre[i](x)

        x = self.fc(x.view(B, -1))
        z = self.norm(x)

        return z, weights
