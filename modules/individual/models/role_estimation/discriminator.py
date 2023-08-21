import torch
import torch.nn as nn

from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Discriminator(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.emb = Embedding(n_pts, config.d_model)
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

        self.adv_layer = nn.Sequential(nn.Linear(config.d_out_feature, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(config.d_out_feature, config.num_classes + 1), nn.Softmax()
        )

    def forward(self, x):
        B = x.size()[0]
        x = self.emb(x)

        x = torch.cat([self.cls.repeat(B, 1, 1), x], dim=1)
        x = self.pe(x)

        for i in range(self.n_tr):
            x, attn = self.trs[i](x)

        x = self.ff(x[:, 0])
        feature = self.selu(x)

        adv = self.adv_layer(feature)
        aux = self.aux_layer(feature)

        return adv, aux, feature, attn
