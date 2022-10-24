import torch.nn as nn

from .activation import Activation


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation):
        super().__init__()

        self.n_heads = n_heads

        self.en_spat = Encoder(d_model, n_heads, d_ff, dropout, activation)
        self.en_temp = Encoder(d_model, n_heads, d_ff, dropout, activation)

        self.de_spat = Decoder(d_model, n_heads, d_ff, dropout, activation)
        self.de_temp = Decoder(d_model, n_heads, d_ff, dropout, activation)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_spat, x_temp, mask_spat=None, mask_temp=None):
        # encoder
        memory_spat, _ = self.en_spat(x_spat)
        memory_temp, _ = self.en_temp(x_temp)

        # repeat mask by n_heads
        if mask_spat is not None:
            shape = mask_spat.size()
            mask_spat = (
                mask_spat.unsqueeze(0)
                .expand(self.n_heads, shape[0], shape[1], shape[2])
                .contiguous()
                .view(self.n_heads * shape[0], shape[1], shape[2])
            )
        if mask_temp is not None:
            shape = mask_temp.size()
            mask_temp = (
                mask_temp.unsqueeze(0)
                .expand(self.n_heads, shape[0], shape[1], shape[2])
                .contiguous()
                .view(self.n_heads * shape[0], shape[1], shape[2])
            )

        # decoder
        x_spat, attn_spat = self.de_spat(memory_spat, memory_temp, mask_spat)
        x_temp, attn_temp = self.de_temp(memory_temp, memory_spat, mask_temp)

        attn_weights = attn_spat * attn_temp.permute(0, 2, 1)

        return x_spat, x_temp, attn_weights


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

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # attention
        x_norm = self.norm1(src)
        attn_out, weights = self.attn(x_norm, x_norm, x_norm)
        src = src + self.dropout1(attn_out)

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
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, src, attn_mask=None):
        # attention1
        x_norm = self.norm1(tgt)
        attn_out, _ = self.attn1(x_norm, x_norm, x_norm)
        tgt = tgt + self.dropout1(attn_out)

        # attention2
        x_norm = self.norm2_1(tgt)
        k = v = self.norm2_2(src)
        attn_out, weights = self.attn2(x_norm, k, v, attn_mask=attn_mask)
        tgt = tgt + self.dropout2(attn_out)

        # feed forward
        x_norm = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(x_norm))

        tgt = self.norm4(tgt)

        return tgt, weights


class FeedFoward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
