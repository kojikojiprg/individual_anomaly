import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "SELU":
            self.activation = nn.SELU(inplace=True)
        else:
            raise NameError

    def forward(self, x):
        return self.activation(x)
