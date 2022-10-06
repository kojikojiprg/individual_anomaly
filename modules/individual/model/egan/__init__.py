import os
from logging import Logger
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator
from .inference import test_discriminator, test_generator
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

    def get_model_paths(self, checkpoint_dir):
        g_path = os.path.join(
            checkpoint_dir,
            f"egan_g_{self._config.model.G.n_sttr}x{self._config.model.G.n_heads}.pth",
        )
        d_path = os.path.join(
            checkpoint_dir,
            f"egan_d_{self._config.model.D.n_sttr}x{self._config.model.D.n_heads}.pth",
        )
        e_path = os.path.join(
            checkpoint_dir,
            f"egan_e_{self._config.model.E.n_sttr}x{self._config.model.E.n_heads}.pth",
        )
        return g_path, d_path, e_path

    def load_checkpoints(self, checkpoint_dir):
        g_path, d_path, e_path = self.get_model_paths(checkpoint_dir)
        if (
            not os.path.exists(g_path)
            or not os.path.exists(d_path)
            or not os.path.exists(e_path)
        ):
            self._logger.warn(f"not exist {g_path}, {d_path} or {e_path}")
            return

        self._logger.info(f"=> loading generator {g_path}")
        param = torch.load(g_path)
        self._G.load_state_dict(param)

        self._logger.info(f"=> loading discriminator {d_path}")
        param = torch.load(d_path)
        self._D.load_state_dict(param)

        self._logger.info(f"=> loading discriminator {e_path}")
        param = torch.load(e_path)
        self._E.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        g_path, d_path, e_path = self.get_model_paths(checkpoint_dir)
        self._logger.info(f"=> saving generator {g_path}")
        torch.save(self._G.state_dict(), g_path)

        self._logger.info(f"=> saving discriminator {d_path}")
        torch.save(self._D.state_dict(), d_path)

        self._logger.info(f"=> saving encoder {e_path}")
        torch.save(self._E.state_dict(), e_path)

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

    def test_generator(self, num_individual: int) -> List[Dict[str, Any]]:
        pass

    def test_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
        pass
