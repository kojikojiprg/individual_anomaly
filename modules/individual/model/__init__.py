import os
from logging import Logger

import torch

from .discriminator import Discriminator
from .generator import Generator
from .train import train


class IndividualGAN:
    def __init__(self, config, device: str, logger: Logger):
        self._config = config
        self._logger = logger
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self.to(device)
        self._device = device

    def to(self, device):
        self._G.to(device)
        self._D.to(device)

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    def load_checkpoints(self, checkpoint_dir):
        g_path = os.path.join(checkpoint_dir, "generator.pth")
        self._logger.info(f"=> loading generator {g_path}")
        param = torch.load(g_path)
        self._G.load_state_dict(param)

        d_path = os.path.join(checkpoint_dir, "discriminator.pth")
        self._logger.info(f"=> loading discriminator {d_path}")
        param = torch.load(d_path)
        self._D.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        g_path = os.path.join(checkpoint_dir, "generator.pth")
        self._logger.info(f"=> saving generator {g_path}")
        torch.save(self._G.state_dict(), g_path)

        d_path = os.path.join(checkpoint_dir, "discriminator.pth")
        self._logger.info(f"=> saving discriminator {d_path}")
        torch.save(self._D.state_dict(), d_path)

    def train(self, dataloader):
        self._G, self._D = train(
            self._G, self._D, dataloader, self._config, self._device, self._logger
        )
