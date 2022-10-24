import torch.nn as nn
from modules.layers.embedding import Embedding
from modules.layers.positional_encoding import PositionalEncoding
from modules.layers.transformer import Encoder as TransformerEncoder


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.en = Encoder(config)
        self.de = Decoder(config)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

    def forward(self, x, mask):
        z, attn = self.en(x, mask)
        x = self.de(z, mask)

        return x, z, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len = config.seq_len * 17
        self.d_model = config.d_model

        self.emb = Embedding(2, config.d_model)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')
        self.pe = PositionalEncoding(config.d_model, self.seq_len)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

        self.tr = nn.ModuleList()
        self.n_tr = config.n_tr_e
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
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

        self.fc = nn.Linear(config.d_model * self.seq_len, config.d_z)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

    def forward(self, x, mask):
        B = x.size()[0]
        x = self.emb(x)
        x = x.view(B, self.seq_len, self.d_model)
        x = self.pe(x)

        mask = mask.view(B, self.seq_len)
        for i in range(self.n_tr):
            x, attn = self.tr[i](x, mask)

        x = x.flatten()
        z = self.fc(x)

        return z, attn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len = config.seq_len * 17
        self.d_model = config.d_model

        self.emb = Embedding(config.d_z, config.d_model * self.seq_len)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

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
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

        self.fc = nn.Linear(config.d_model, 2)
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'TRAINABLE PARAMS: {params}')

    def forward(self, z, mask):
        B = z.size()[0]

        x = self.emb(z)
        x = z.view(B, self.seq_len, self.d_model)

        mask = mask.view(B, self.seq_len)
        for i in range(self.n_tr):
            x, _ = self.tr[i](x, mask)
        x = self.fc(x)

        return x
