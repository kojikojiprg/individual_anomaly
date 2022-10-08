import os
from logging import Logger
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator
from .inference import infer_discriminator, infer_generator
from .train import train


class IndividualEGAN(nn.Module):
    def __init__(self, config, device: str, logger: Logger):
        super().__init__()

        self._config = config
        self._logger = logger
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._E = Encoder(config.model.E)
        self.to(device)
        self._device = device

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    @property
    def Encoder(self):
        return self._E

    def get_model_path(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "egan.pth")
        return path

    def load_checkpoints(self, checkpoint_dir):
        path = self.get_model_path(checkpoint_dir)
        self._logger.info(f"=> loading EGAN parameters {path}")
        param = torch.load(path)
        self.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        path = self.get_model_path(checkpoint_dir)
        self._logger.info(f"=> saving EGAN parameters {path}")
        torch.save(self.state_dict(), path)

    def train(self, train_dataloader):
        self._G, self._D, self._E = train(
            self._G,
            self._D,
            self._E,
            train_dataloader,
            self._config,
            self._device,
            self._logger,
        )

    def infer_generator(self, num_individual: int) -> List[Dict[str, Any]]:
        results = infer_generator(
            self._G, num_individual, self._config.model.G.d_z, self._device
        )
        return results

    def infer_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
        results = infer_discriminator(self._D, test_dataloader)
        return results
