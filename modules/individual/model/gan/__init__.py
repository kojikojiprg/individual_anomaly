import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .generator import Generator


class IndividualGAN(LightningModule):
    def __init__(self, config, data_type):
        super().__init__()

        self._config = config
        self._data_type = data_type
        self._G = Generator(config.model.G, data_type)
        self._D = Discriminator(config.model.D, data_type)
        self._d_z = config.model.G.d_z
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="gan_gloss_{epoch}",
                monitor="g_loss",
                mode="max",
            ),
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="gan_dloss_{epoch}",
                monitor="d_loss",
                mode="min",
                save_last=True,
            ),
        ]

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, keypoints = batch
        mini_batch_size = keypoints.size()[0]

        # make random noise
        z = torch.randn(mini_batch_size, self._d_z).to(self.device)

        if optimizer_idx == 0:
            # train Generator
            fake_keypoints, _ = self._G(z)
            d_out_fake, _, _, _ = self._D(fake_keypoints)

            g_loss = -d_out_fake.mean()
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss

        if optimizer_idx == 1:
            # train Discriminator
            d_out_real, _, _, _ = self._D(keypoints)

            fake_keypoints, _ = self._G(z)
            d_out_fake, _, _, _ = self._D(fake_keypoints)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, keypoints = batch
        out = self._D(keypoints)
        if dataloader_idx == 0:
            # train data
            return {"stage": "train", "frame_num": frame_nums, "pid": pids, "pred": out}
        if dataloader_idx == 1:
            # train data
            return {"stage": "test", "frame_num": frame_nums, "pid": pids, "pred": out}

    def configure_optimizers(self):
        g_optim = torch.optim.Adam(
            self._G.parameters(),
            self._config.optim.lr_g,
            (self._config.optim.beta1, self._config.optim.beta2),
        )
        d_optim = torch.optim.Adam(
            self._D.parameters(),
            self._config.optim.lr_d,
            (self._config.optim.beta1, self._config.optim.beta2),
        )

        return [g_optim, d_optim], []
