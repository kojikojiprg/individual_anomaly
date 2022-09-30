import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()

        self.d_model = d_model
        self.length = length

        pe = torch.zeros(length, d_model)
        for pos in range(length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        self._pe = pe.unsqueeze(0)
        self._pe.requires_grad = False

    def to(self, device):
        self._pe = self._pe.to(device)

    def forward(self, x):
        return x + self._pe
