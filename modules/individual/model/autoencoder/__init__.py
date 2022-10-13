import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .decoder import Decoder
from .encoder import Encoder


class IndividualAutoencoder(LightningModule):
    def __init__(self, config, data_type):
        super().__init__()

        self._config = config
        self._data_type = data_type
        self._E = Encoder(config.model.E, data_type)
        self._D = Decoder(config.model.D, data_type)
        self._criterion = torch.nn.MSELoss()
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"autoencoder_{data_type}_" + "{loss:.2f}_{epoch}",
                monitor="loss",
                save_last=True,
            ),
            EarlyStopping(monitor="loss"),
        ]
        self._callbacks[0].CHECKPOINT_NAME_LAST = (
            f"autoencoder_{data_type}_" + "{loss:.2f}_{epoch}"
        )

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

        self.log("loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, keypoints = batch
        out = self(keypoints)

        if dataloader_idx == 0:
            # train data
            return {"stage": "train", "frame_num": frame_nums, "pid": pids, "pred": out}
        if dataloader_idx == 1:
            # train data
            return {"stage": "test", "frame_num": frame_nums, "pid": pids, "pred": out}

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            self._config.optim.lr,
            (self._config.optim.beta1, self._config.optim.beta2),
        )
        return optim
