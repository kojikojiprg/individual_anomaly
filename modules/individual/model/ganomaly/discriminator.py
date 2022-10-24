import torch
import torch.nn as nn
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len = config.seq_len * 17 + 1  # add [cls] token
        self.d_model = config.d_model

        self.emb = Embedding(2, config.d_model)
        self.pe = PositionalEncoding(config.d_model, self.seq_len)
        self.adversarial_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        self.tr = nn.ModuleList()
        self.n_tr = config.n_tr
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

        self.out = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        B = x.size()[0]

        # embedding
        x = self.emb(x)

        # add adversarial token
        adv_tkn_batch = self.adversarial_token.repeat(B)
        x = torch.cat([adv_tkn_batch, x], dim=1)

        # positional encoding
        x = x.view(B, self.seq_len, self.d_model)
        x = self.pe(x)

        for i in range(self.n_tr):
            x, attn = self.tr[i](x, mask)

        # extract adversarial attention weights and feature
        adv_attn = attn[:, 0]
        adv_tkn_feature = x.view(B, self.seq_len, self.d_model)[:, 0]

        # predict real or fake
        pred = self.out(adv_tkn_feature)

        return pred, adv_tkn_feature, adv_attn
