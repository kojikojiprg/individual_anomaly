import os
import time
from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from modules.individual.format import Format

from .decoder import Decoder
from .encoder import Encoder


class IndividualAutoEncoder(nn.Module):
    def __init__(self, config, device: str, logger: Logger):
        super().__init__()

        self._config = config
        self._logger = logger
        self._E = Encoder(config.model.E)
        self._D = Decoder(config.model.D)
        self.to(device)
        self._device = device

    @property
    def Decoder(self):
        return self._D

    @property
    def Encoder(self):
        return self._E

    def forward(self, x):
        z, feature, weights_spat, weights_temp = self._E(x)
        y = self._D(z)
        return y, z, feature, weights_spat, weights_temp

    def get_model_path(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "autoencoder.pth")
        return path

    def load_checkpoints(self, checkpoint_dir):
        path = self.get_model_path(checkpoint_dir)
        self._logger.info(f"=> loading AutoEncoder parameters {path}")
        param = torch.load(path)
        self.load_state_dict(param)

    def save_checkpoints(self, checkpoint_dir):
        path = self.get_model_path(checkpoint_dir)
        self._logger.info(f"=> saving AutoEncoder parameters {path}")
        torch.save(self.state_dict(), path)

    def train(self, train_dataloader):
        n_epochs = self._config.epochs

        optim = torch.optim.Adam(
            self.parameters(),
            self._config.optim.lr,
            (self._config.optim.beta1, self._config.optim.beta2),
        )

        criterion = nn.MSELoss().to(self._device)

        super().train()
        torch.backends.cudnn.benchmark = True

        self._logger.info("=> start training")
        for epoch in range(n_epochs):
            losses = []

            t_epoch_start = time.time()
            for frame_nums, pids, keypoints in train_dataloader:
                keypoints = keypoints.to(self._device)

                y, _, _, _, _ = self(keypoints)
                loss = criterion(keypoints, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

            t_epoch_finish = time.time()
            loss_mean = np.average(losses)
            self._logger.info("-------------")
            self._logger.info(f"Epoch {epoch + 1}/{n_epochs} | loss:{loss_mean:.5f}")
            self._logger.info(
                "time: {:.3f} sec.".format(t_epoch_finish - t_epoch_start)
            )
        self._logger.info("=> finish training")

    def infer(
        self, test_dataloader: torch.utils.data.DataLoader
    ) -> List[Dict[str, Any]]:
        results = []
        super().eval()
        with torch.no_grad():
            for frame_nums, ids, kps in test_dataloader:
                _, features, weights_spat, weights_temp = self(kps)

                for frame_num, i, feature, w_spat, w_temp in zip(
                    frame_nums, ids, features, weights_spat, weights_temp
                ):
                    results.append(
                        {
                            Format.frame_num: frame_num,
                            Format.id: i,
                            Format.keypoints: kps,
                            Format.feature: feature,
                            Format.w_spat: w_spat,
                            Format.w_temp: w_temp,
                        }
                    )

        return results
