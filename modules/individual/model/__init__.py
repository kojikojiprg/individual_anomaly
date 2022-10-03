import os
from logging import Logger
from typing import Any, Dict, List

import torch

from .layers.discriminator import Discriminator
from .layers.generator import Generator
from .test import test_discriminator, test_generator
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

    def get_model_paths(self, checkpoint_dir):
        g_path = os.path.join(
            checkpoint_dir,
            f"generator_{self._config.model.G.n_sttr}x{self._config.model.G.n_heads}.pth",
        )
        d_path = os.path.join(
            checkpoint_dir,
            f"discriminator_{self._config.model.D.n_sttr}x{self._config.model.D.n_heads}.pth",
        )
        return g_path, d_path

    def load_checkpoints(self, checkpoint_dir):
        g_path, d_path = self.get_model_paths(checkpoint_dir)
        if not os.path.exists(g_path) or not os.path.exists(d_path):
            self._logger.warn(f"not exist {g_path} or {d_path}")
            return

        self._logger.info(f"=> loading generator {g_path}")
        param = torch.load(g_path)
        self._G.load_state_dict(param)

        self._logger.info(f"=> loading discriminator {d_path}")
        param = torch.load(d_path)
        self._D.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        g_path, d_path = self.get_model_paths(checkpoint_dir)
        self._logger.info(f"=> saving generator {g_path}")
        torch.save(self._G.state_dict(), g_path)

        self._logger.info(f"=> saving discriminator {d_path}")
        torch.save(self._D.state_dict(), d_path)

    def train(self, train_dataloader):
        self._G, self._D = train(
            self._G, self._D, train_dataloader, self._config, self._device, self._logger
        )

    def test_generator(self, num_individual: int) -> List[Dict[str, Any]]:
        results = test_generator(
            self._G, num_individual, self._config.model.G.d_z, self._device
        )
        return results

    def test_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
        results = test_discriminator(self._D, test_dataloader)
        return results
