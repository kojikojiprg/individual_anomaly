import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length, device):
        super().__init__()

        self.d_model = d_model
        self.length = length

        pe = torch.zeros(length, d_model)
        for pos in range(length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
        self.pe = self.pe.to(device)

    def forward(self, x):
        return x + self.pe
