import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .decoder import Decoder
from .encoder import Encoder


class IndividualAutoencoder(LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._E = Encoder(config.model.E)
        self._D = Decoder(config.model.D)
        self._criterion = nn.MSELoss()
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="{epoch}-loss{loss:.5f}",
                monitor="loss",
            )
        ]

    @property
    def Decoder(self):
        return self._D

    @property
    def Encoder(self):
        return self._E

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def forward(self, x):
        z, feature, weights_spat, weights_temp = self._E(x)
        y = self._D(z)
        return y, z, feature, weights_spat, weights_temp

    def training_step(self, batch, batch_idx):
        frame_nums, pids, keypoints = batch

        y, _, _, _, _ = self(keypoints)
        loss = self._criterion(keypoints, y)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            self._config.optim.lr,
            (self._config.optim.beta1, self._config.optim.beta2),
        )
        return optim

    # def infer(
    #     self, test_dataloader: torch.utils.data.DataLoader
    # ) -> List[Dict[str, Any]]:
    #     results = []
    #     super().eval()
    #     with torch.no_grad():
    #         for frame_nums, ids, kps in test_dataloader:
    #             _, features, weights_spat, weights_temp = self(kps)

    #             for frame_num, i, feature, w_spat, w_temp in zip(
    #                 frame_nums, ids, features, weights_spat, weights_temp
    #             ):
    #                 results.append(
    #                     {
    #                         IndividualDataFormat.frame_num: frame_num,
    #                         IndividualDataFormat.id: i,
    #                         IndividualDataFormat.keypoints: kps,
    #                         IndividualDataFormat.feature: feature,
    #                         IndividualDataFormat.w_spat: w_spat,
    #                         IndividualDataFormat.w_temp: w_temp,
    #                     }
    #                 )

    #     return results
