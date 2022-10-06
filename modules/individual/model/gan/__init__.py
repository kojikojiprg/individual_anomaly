import os
from logging import Logger
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator
from .inference import infer_discriminator, infer_generator
from .train import train


class IndividualGAN(nn.Module):
    def __init__(self, config, device: str, logger: Logger):
        super().__init__()

        self._config = config
        self._logger = logger
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self.to(device)
        self._device = device

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    def get_model_path(self, checkpoint_dir):
        path = os.path.join(
            checkpoint_dir,
            f"gan_g{self._config.model.G.n_sttr}d{self._config.model.D.n_sttr}.pth",
        )
        return path

    def load_checkpoints(self, checkpoint_dir):
        path = self.get_model_paths(checkpoint_dir)
        self._logger.info(f"=> loading GAN parameters {path}")
        param = torch.load(path)
        self.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        path = self.get_model_paths(checkpoint_dir)
        self._logger.info(f"=> saving GAN parameters {path}")
        torch.save(self.state_dict(), path)

    def train(self, train_dataloader):
        self._G, self._D = train(
            self._G, self._D, train_dataloader, self._config, self._device, self._logger
        )

    def test_generator(self, num_individual: int) -> List[Dict[str, Any]]:
        results = infer_generator(
            self._G, num_individual, self._config.model.G.d_z, self._device
        )
        return results

    def test_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
        results = infer_discriminator(self._D, test_dataloader)
        return results
