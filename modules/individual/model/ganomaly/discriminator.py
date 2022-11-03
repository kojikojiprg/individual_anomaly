import torch
import torch.nn as nn
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = Embedding(config.n_kps, config.d_model)
        self.pe = PositionalEncoding(config.d_model, config.seq_len + 1)
        self.cls = nn.Parameter(torch.randn((1, 1, config.d_model)))

        self.trs = nn.ModuleList()
        self.n_tr = config.n_tr
        for _ in range(self.n_tr):
            self.trs.append(
                TransformerEncoder(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
            )

        self.ff = nn.Linear(config.d_model, config.d_out_feature)
        self.selu = nn.SELU(inplace=True)
        self.out = nn.Linear(config.d_out_feature, 1)

    def forward(self, x, mask):
        B = x.size()[0]
        x = self.emb(x)

        x = torch.cat([self.cls.repeat(B, 1, 1), x], dim=1)
        x = self.pe(x)

        for i in range(self.n_tr):
            x, _ = self.trs[i](x)

        x = self.ff(x[:, 0])
        feature = self.selu(x)
        pred = self.out(feature)

        return pred, feature
