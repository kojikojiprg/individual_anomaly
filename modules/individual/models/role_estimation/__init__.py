from types import SimpleNamespace

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from modules.visualize.individual import plot_val_kps

from .discriminator import Discriminator
from .generator import Generator


class RoleEstimation(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        data_type: str,
        prediction_type: str = None,
    ):
        super().__init__()

        self._config = config
        self._num_classes = self._config.model.D.num_classes
        self._prediction_type = prediction_type

        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._l_adv = torch.nn.BCELoss()
        self._l_lat = torch.nn.MSELoss()
        self._l_aux = torch.nn.CrossEntropyLoss()

        self._seq_len = config.dataset.seq_len
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"re_{data_type}_seq{self._seq_len}_dmin",
                monitor="d_loss",
                mode="min",
                save_last=True,
            ),
        ]

        last_name = f"re_{data_type}_seq{self._seq_len}_last"
        self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

        self._random_generator = None

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def generate_noise(self, batch_size):
        z = torch.randn(
            (batch_size, self._config.model.G.d_z), generator=self._random_generator
        ).to(self.device)
        return z

    def on_train_start(self):
        self._random_generator = torch.random.manual_seed(self.device.index + 1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, bbox, kps, aux_real_gt = batch
        batch_size = len(frame_nums)

        # make ground truth
        adv_real_gt = torch.full((batch_size,), 0.9).to(self.device)
        adv_fake_gt = torch.zeros((batch_size,), dtype=torch.float32).to(self.device)
        aux_real_gt = torch.nn.functional.one_hot(
            aux_real_gt, num_classes=self._num_classes + 1
        ).float()
        aux_fake_gt = torch.Tensor(
            np.full(
                (batch_size, self._num_classes + 1),
                np.eye(self._num_classes + 1)[self._num_classes],
                dtype=np.float32,
            )
        ).to(self.device)

        # pred generator
        z = self.generate_noise(batch_size)
        fake, attn_g = self._G(z)
        adv_fake, aux_fake, f_d_fake, attn_d_fake = self._D(fake)

        # pred discriminator
        x = torch.cat([bbox, kps], dim=2)
        adv_real, aux_real, f_d_real, attn_d_real = self._D(x)

        if optimizer_idx == 0:
            g_loss = self._l_adv(adv_fake.view(-1), adv_real_gt) + self._l_lat(
                f_d_fake.view(-1), f_d_real.view(-1)
            )
            self.log_dict(
                {
                    "g_loss": g_loss,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            return g_loss
        if optimizer_idx == 1:
            l_real = (
                self._l_adv(adv_real.view(-1), adv_real_gt)
                + self._l_aux(aux_real, aux_real_gt)
            ) / 2
            l_fake = (
                self._l_adv(adv_fake.view(-1), adv_fake_gt)
                + self._l_aux(aux_fake, aux_fake_gt)
            ) / 2
            d_loss = l_real + l_fake
            self.log_dict(
                {
                    "d_loss": d_loss,
                    # "d_l_real": l_real,
                    # "d_l_fake": l_fake,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            return d_loss

    def validation_step(self, batch, batch_idx):
        frame_nums, pids, bbox_real, kps_real, aux_real_gt = batch
        batch_size = len(frame_nums)

        # make true data
        adv_real_gt = np.ones((batch_size,), dtype=np.float32)
        adv_fake_gt = np.zeros((batch_size,), dtype=np.float32)
        aux_real_gt = (
            torch.nn.functional.one_hot(aux_real_gt, num_classes=self._num_classes + 1)
            .cpu()
            .numpy()
        )
        aux_fake_gt = np.full(
            (batch_size, self._num_classes + 1),
            np.eye(self._num_classes + 1)[self._num_classes],
            dtype=np.float32,
        )

        # pred generator
        z = self.generate_noise(batch_size)
        fake, attn_g = self._G(z)
        adv_fake, aux_fake, f_d_fake, attn_d_fake = self._D(fake)
        _, kps_fake = fake[:, :, :2], fake[:, :, 2:]

        # pred discriminator
        x = torch.cat([bbox_real, kps_real], dim=2)
        adv_real, aux_real, f_d_real, attn_d_real = self._D(x)

        # adv auc
        adv_real = adv_real.view(-1).cpu().numpy()
        adv_fake = adv_fake.view(-1).cpu().numpy()
        adv_gt = np.concatenate([adv_real_gt, adv_fake_gt])
        adv = np.concatenate([adv_real, adv_fake])
        adv_auc = roc_auc_score(adv_gt, adv)

        # aux auc
        aux_real = aux_real.cpu().numpy()
        aux_fake = aux_fake.cpu().numpy()
        aux_gt = np.concatenate([aux_real_gt, aux_fake_gt], axis=0)
        aux = np.concatenate([aux_real, aux_fake], axis=0)
        aux_auc = roc_auc_score(aux_gt, aux)

        self.log_dict({"adv_auc": adv_auc, "aux_auc": aux_auc})
        return pids, kps_real.cpu().numpy(), kps_fake.cpu().numpy()

    def validation_epoch_end(self, outputs):
        if self.current_epoch == 0 or self.current_epoch % 9 == 0:
            out = outputs[0]
            plot_val_kps(
                out[1][0],  # dmy
                out[2][0],
                out[0][0],
                self.current_epoch,
                "role_estimation",
                "",
            )

    def predict_step(self, batch, batch_idx, dataloader_idx):
        pass

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

        epochs = self._config.train.epochs
        m50 = int(epochs * 0.5)
        m70 = int(epochs * 0.7)
        m90 = int(epochs * 0.9)
        g_sdl = torch.optim.lr_scheduler.MultiStepLR(
            g_optim, [m50, m70, m90], self._config.optim.lr_rate_g
        )
        d_sdl = torch.optim.lr_scheduler.MultiStepLR(
            d_optim, [m50, m70, m90], self._config.optim.lr_rate_d
        )

        return [g_optim, d_optim], [g_sdl, d_sdl]
