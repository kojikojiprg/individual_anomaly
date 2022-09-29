import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_input, d_emb, d_output):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_input, d_emb),
            nn.Linear(d_emb, d_output),
        )

    def forward(self, x):
        return self.linear(x)
