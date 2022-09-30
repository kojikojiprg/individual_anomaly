import torch.nn as nn


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()

        self.en_spat = Encoder(d_model, n_heads, d_ff, dropout, activation)
        self.en_temp = Encoder(d_model, n_heads, d_ff, dropout, activation)

        self.de_spat = Decoder(d_model, n_heads, d_ff, dropout, activation)
        self.de_temp = Decoder(d_model, n_heads, d_ff, dropout, activation)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_spat, x_temp):
        # encoder
        memory_spat, _ = self.en_spat(x_spat)
        memory_temp, _ = self.en_temp(x_temp)

        # decoder
        x_spat, weights_spat = self.de_spat(memory_spat, memory_temp)
        x_temp, weights_temp = self.de_temp(memory_temp, memory_spat)

        return x_spat, x_temp, weights_spat, weights_temp


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = FeedFoward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # attention
        x_norm = self.norm1(src)
        attn, weights = self.attn(x_norm, x_norm, x_norm)
        src = src + self.dropout1(attn)

        # feed forward
        x_norm = self.norm2(src)
        src = src + self.dropout2(self.ff(x_norm))

        src = self.norm3(src)

        return src, weights


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()

        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = FeedFoward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, src):
        # attention1
        x_norm = self.norm1(tgt)
        attn, _ = self.attn1(x_norm, x_norm, x_norm)
        tgt = tgt + self.dropout1(attn)

        # attention2
        x_norm = self.norm2_1(tgt)
        k = v = self.norm2_2(src)
        attn, weights = self.attn2(x_norm, k, v)
        tgt = tgt + self.dropout2(attn)

        # feed forward
        x_norm = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(x_norm))

        tgt = self.norm4(tgt)

        return tgt, weights


class FeedFoward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

    @staticmethod
    def _get_activation(activation):
        if activation == "ReLU":
            return nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == "GELU":
            return nn.GELU()
        elif activation == "SELU":
            return nn.SELU(inplace=True)
        else:
            raise NameError
