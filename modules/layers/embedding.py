import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()

        self.linear = nn.Linear(d_input, d_output)

        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # stop updating params
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x)
