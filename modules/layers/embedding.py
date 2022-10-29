import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)  # embed vertex (x, y)
        self.conv2 = nn.Conv2d(d_in, d_model, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, N, _, _ = x.size()  # (batch_len, data_len, d_in, 2)

        # conv vertex (x, y)
        x = x.permute(0, 3, 1, 2)  # (batch_len, 2, data_len, d_in)
        x = self.conv1(x)  # (batch_len, 1, data_len, d_in)

        # conv data
        x = x.permute(0, 3, 2, 1)  # (batch_len, d_in, data_len, 1)
        x = self.conv2(x)  # (batch_len, d_model, data_len, 1)

        x = x.permute(0, 2, 1, 3).view(B, N, -1)  # (batch_size, data_len, d_model)

        return x
