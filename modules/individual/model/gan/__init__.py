import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .generator import Generator


class IndividualGAN(LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._d_z = config.model.G.d_z
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="{epoch}-loss{g_loss:.5f}",
                monitor="g_loss",
                mode="max",
            ),
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="{epoch}-loss{d_loss:.5f}",
                monitor="d_loss",
                mode="min",
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
        device = keypoints.get_device()

        # make random noise
        z = torch.randn(mini_batch_size, self._d_z).to(device)

        if optimizer_idx == 0:
            # train Generator
            fake_keypoints, _ = self._G(z)
            d_out_fake, _, _, _ = self._D(fake_keypoints)

            g_loss = -d_out_fake.mean()
            self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=True)
            return g_loss

        if optimizer_idx == 1:
            # train Discriminator
            d_out_real, _, _, _ = self._D(keypoints)

            fake_keypoints, _ = self._G(z)
            d_out_fake, _, _, _ = self._D(fake_keypoints)

            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True)
            return d_loss

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

    # def infer_generator(self, num_individual: int) -> List[Dict[str, Any]]:
    #     results = infer_generator(
    #         self._G, num_individual, self._config.model.G.d_z, self._device
    #     )
    #     return results

    # def infer_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
    #     results = infer_discriminator(self._D, test_dataloader)
    #     return results
