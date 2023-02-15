import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        self._pe = torch.zeros(length, d_model)
        self._w = nn.Parameter(torch.randn(length))
        for pos in range(0, length, 2):
            self._pe[pos, :] = math.sin(pos * self._w[pos])
            if pos + 1 == len(self._pe):
                break  # d_model is odd number
            self._pe[pos + 1, :] = math.cos(pos * self._w[pos + 1])

        self._pe = (self._pe.unsqueeze(0) + 1) / 2  # [-1, 1] -> [0, 1]
        self.register_buffer("pe", self._pe)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return x + self.pe
