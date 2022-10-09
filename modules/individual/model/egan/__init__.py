import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator


class IndividualEGAN(LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._E = Encoder(config.model.E)
        self._d_z = config.model.G.d_z
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
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
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="{epoch}-loss{d_loss:.5f}",
                monitor="e_loss",
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
    def Encoder(self):
        return self._E

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, keypoints = batch
        mini_batch_size = keypoints.size()[0]
        device = keypoints.get_device()

        # make random noise
        z = torch.randn(mini_batch_size, self._d_z).to(device)

        # make true data
        label_real = torch.full((mini_batch_size,), 1, dtype=torch.float32).to(device)
        label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float32).to(device)

        if optimizer_idx == 0:
            # train Generator
            z = torch.randn(mini_batch_size, self._d_z).to(device)
            fake_keypoints, _ = self._G(z)
            d_out_fake, _ = self._D(fake_keypoints, z)

            g_loss = self._criterion(d_out_fake.view(-1), label_real)
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss

        if optimizer_idx == 1:
            # train Discriminator
            z_out_real, _, _, _ = self._E(keypoints)
            d_out_real, _ = self._D(keypoints, z_out_real)

            z = torch.randn(mini_batch_size, self._d_z).to(device)
            fake_keypoints, _ = self._G(z)
            d_out_fake, _ = self._D(fake_keypoints, z)

            d_loss_real = self._criterion(d_out_real.view(-1), label_real)
            d_loss_fake = self._criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

        if optimizer_idx == 2:
            # train Encoder
            z_out_real, _, _, _ = self._E(keypoints)
            d_out_real, _ = self._D(keypoints, z_out_real)

            e_loss = self._criterion(d_out_real.view(-1), label_fake)
            self.log("e_loss", e_loss, prog_bar=True, on_step=True)
            return e_loss

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
        e_optim = torch.optim.Adam(
            self._E.parameters(),
            self._config.optim.lr_e,
            (self._config.optim.beta1, self._config.optim.beta2),
        )

        return [g_optim, d_optim, e_optim], []

    # def infer_generator(self, num_individual: int) -> List[Dict[str, Any]]:
    #     results = infer_generator(
    #         self._G, num_individual, self._config.model.G.d_z, self._device
    #     )
    #     return results

    # def infer_discriminator(self, test_dataloader) -> List[Dict[str, Any]]:
    #     results = infer_discriminator(self._D, test_dataloader)
    #     return results
