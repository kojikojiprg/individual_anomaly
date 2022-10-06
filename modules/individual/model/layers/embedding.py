import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()

        self.linear = nn.Linear(d_input, d_output)
        self.linear.requires_grad_ = False

    def forward(self, x):
        return self.linear(x)
