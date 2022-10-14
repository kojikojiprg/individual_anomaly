import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()

        pe = torch.zeros(length, d_model)
        for pos in range(length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        self._pe = pe.unsqueeze(0)
        self._pe.requires_grad = False
        self.register_buffer("pe", self._pe)

    def forward(self, x):
        return x + self.pe
