from types import SimpleNamespace

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from modules import DataFormat

from .discriminator import Discriminator
from .generator import Generator


class GanomalyKps(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        data_type: str,
        masking: bool,
    ):
        super().__init__()
        # self.automatic_optimization = False

        self._config = config
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._data_type = data_type
        self._masking = masking

        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._l_adv = torch.nn.BCEWithLogitsLoss()
        self._l_con = torch.nn.MSELoss()
        self._l_lat = torch.nn.MSELoss()

        seq_len = config.dataset.seq_len
        masked_str = ""
        if masking:
            masked_str = "_masked"
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"ganomaly{masked_str}_{data_type}_seq{seq_len}_gcon_min",
                monitor="g_l_con",
                mode="min",
                save_last=True,
            ),
        ]

        last_name = f"ganomaly{masked_str}_{data_type}_seq{seq_len}_last"
        self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def forward(self, kps_real, mask):
        if not self._masking:
            mask = None

        # pred real
        pred_real, f_real, attn_d_real = self._D(kps_real, mask)

        # pred fake
        kps_fake, z, attn_ge, attn_gd = self._G(kps_real, mask)
        pred_fake, f_fake, attn_d_fake = self._D(kps_fake, mask)

        return pred_real, pred_fake, kps_fake, z, attn_ge, f_real, f_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, kps_real, mask = batch
        batch_size = kps_real.size()[0]

        # make true data
        label_real = torch.ones((batch_size,), dtype=torch.float32).to(self.device)
        label_fake = torch.zeros((batch_size,), dtype=torch.float32).to(self.device)

        # predict
        pred_real, pred_fake, kps_fake, _, _, f_real, f_fake = self(kps_real, mask)

        if optimizer_idx == 0:
            # generator loss
            l_adv = (
                self._l_adv(pred_fake.view(-1), label_real) * self._config.loss.G.w_adv
            )
            l_con = self._l_con(kps_fake, kps_real) * self._config.loss.G.w_con
            l_lat = self._l_lat(f_fake, f_real) * self._config.loss.G.w_lat
            g_loss = l_adv + l_con + l_lat
            # self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            self.log_dict(
                {
                    "g_loss": g_loss,
                    "g_l_adv": l_adv,
                    "g_l_con": l_con,
                    "g_l_lat": l_lat,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            return g_loss

        if optimizer_idx == 1:
            # discriminator loss
            l_adv_real = self._l_adv(pred_real.view(-1), label_real)
            l_adv_fake = self._l_adv(pred_fake.view(-1), label_fake)
            d_loss = l_adv_real + l_adv_fake
            # self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            self.log_dict(
                {
                    "d_loss": d_loss,
                    "d_l_adv_real": l_adv_real,
                    "d_l_adv_fake": l_adv_fake,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )
            return d_loss

    def validation_step(self, batch, batch_idx):
        _, _, kps_real, mask = batch
        batch_size = kps_real.size()[0]

        # make label
        label_real = np.ones((batch_size,), dtype=np.float32)
        label_fake = np.zeros((batch_size,), dtype=np.float32)

        # predict
        pred_real, pred_fake, _, _, _, _, _ = self(kps_real, mask)
        pred_real = pred_real.view(-1).cpu().numpy()
        pred_fake = pred_fake.view(-1).cpu().numpy()

        label = np.concatenate([label_real, label_fake])
        pred = np.concatenate([pred_real, pred_fake])
        d_auc = roc_auc_score(label, pred)
        self.log("d_auc", d_auc)

    # def validation_epoch_end(self, outputs):
    #     for out in outputs:
    #         pids, kps_real, kps_fake, z, attn = out
    #         for i in range(len(kps_real)):
    #             plot_val_kps(
    #                 kps_real[i],
    #                 kps_fake[i],
    #                 pids[i],
    #                 self.current_epoch,
    #                 self._data_type,
    #             )

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    def anomaly_score(self, kps_real, kps_fake, mask_bool, f_real, f_fake):
        B, T = kps_real.size()[:2]

        kps_real = kps_real.view(B, T, 17, 2)
        kps_fake = kps_fake.view(B, T, 17, 2)

        # apply mask
        if self._masking:
            kps_real[mask_bool] = 0.0
            kps_fake[mask_bool] = 0.0

        # calc the difference between real keypoints and fake keypoints
        loss_residual = torch.abs(kps_real - kps_fake)
        loss_residual = loss_residual.view(B, -1)
        loss_residual = torch.mean(loss_residual, dim=1)

        # calc the absolute difference between real feature and fake feature
        loss_discrimination = torch.abs(f_real - f_fake)
        loss_discrimination = loss_discrimination.view(B, -1)
        loss_discrimination = torch.mean(loss_discrimination, dim=1)

        return loss_residual, loss_discrimination

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, kps_real, mask = batch
        mask_bool = mask < 0

        # predict
        _, _, kps_fake, z, attn, f_real, f_fake = self(kps_real, mask)

        l_resi, l_disc = self.anomaly_score(
            kps_real,
            kps_fake,
            mask_bool,
            f_real,
            f_fake,
        )

        preds = []
        for i in range(len(frame_nums)):
            preds.append(
                {
                    DataFormat.frame_num: int(self._to_numpy(frame_nums[i])),
                    DataFormat.id: int(pids[i]),
                    # DataFormat.kps_real: self._to_numpy(kps_real[i]),
                    DataFormat.kps_fake: self._to_numpy(kps_fake[i]),
                    DataFormat.z: self._to_numpy(z[i]),
                    # DataFormat.attn: self._to_numpy(attn[i]),
                    # DataFormat.f_real: self._to_numpy(f_real[i]),
                    # DataFormat.f_fake: self._to_numpy(f_fake[i]),
                    DataFormat.loss_r: self._to_numpy(l_resi[i]),
                    DataFormat.loss_d: self._to_numpy(l_disc[i]),
                }
            )
        return preds

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
