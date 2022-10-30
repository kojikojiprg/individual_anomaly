import numpy as np
import torch
from modules.individual import IndividualDataFormat, IndividualDataTypes
from modules.visualize.individual import plot_val_kps
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from .discriminator import Discriminator
from .generator import Generator


class IndividualGanomaly(LightningModule):
    def __init__(self, config, data_type):
        super().__init__()

        if data_type == IndividualDataTypes.both:
            self.n_kps = 34  # both
        else:
            self.n_kps = 17  # abs or rel

        self._config = config
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._data_type = data_type

        self._G = Generator(config.model.G, data_type)
        self._D = Discriminator(config.model.D, data_type)
        self._l_adv = torch.nn.BCEWithLogitsLoss()
        self._l_con = torch.nn.MSELoss()
        self._l_lat = torch.nn.MSELoss()

        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"ganomaly_{data_type}_" + "{d_loss:.2f}_{epoch}",
                monitor="d_loss",
                mode="min",
                save_last=True,
            ),
        ]
        self._callbacks[0].CHECKPOINT_NAME_LAST = f"ganomaly_last-{data_type}"

    @property
    def Generator(self):
        return self._G

    @property
    def Discriminator(self):
        return self._D

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def forward(self, kps_real, mask_batch):
        # predict Z and fake keypoints
        kps_fake, z, attn_g = self._G(kps_real)

        # extract feature maps of real keypoints and fake keypoints from D
        _, f_real = self._D(kps_real, mask_batch)
        _, f_fake = self._D(kps_fake, mask_batch)

        return kps_fake, z, attn_g, f_real, f_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, kps_real, mask = batch
        batch_size = kps_real.size()[0]

        # make true data
        label_real = torch.ones((batch_size,), dtype=torch.float32).to(self.device)
        label_fake = torch.zeros((batch_size,), dtype=torch.float32).to(self.device)

        # pred real
        pred_real, feature_real = self._D(kps_real, mask)

        # pred fake
        kps_fake, _, _ = self._G(kps_real)
        pred_fake, feature_fake = self._D(kps_fake, mask)

        if optimizer_idx == 0:
            # generator loss
            l_adv = (
                self._l_adv(pred_fake.view(-1), label_real) * self._config.loss.G.w_adv
            )
            l_con = self._l_con(kps_fake, kps_real) * self._config.loss.G.w_con
            l_lat = self._l_lat(feature_fake, feature_real) * self._config.loss.G.w_lat
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
        frame_nums, pids, kps_real, mask = batch
        batch_size = kps_real.size()[0]

        # make true data
        label_real = np.ones((batch_size,), dtype=np.float32)
        label_fake = np.zeros((batch_size,), dtype=np.float32)

        # pred real
        pred_real, _ = self._D(kps_real, mask)

        # pred fake
        kps_fake, z, attn = self._G(kps_real)
        pred_fake, _ = self._D(kps_fake, mask)

        kps_real = kps_real.cpu().numpy()
        kps_fake = kps_fake.cpu().numpy()
        z = z.cpu().numpy()
        attn = attn.cpu().numpy()
        pred_real = pred_real.view(-1).cpu().numpy()
        pred_fake = pred_fake.view(-1).cpu().numpy()

        label = np.concatenate([label_real, label_fake])
        pred = np.concatenate([pred_real, pred_fake])
        d_auc = roc_auc_score(label, pred)
        self.log("d_auc", d_auc)

        return pids, kps_real, kps_fake, z, attn

    def validation_epoch_end(self, outputs):
        for out in outputs:
            pids, kps_real, kps_fake, z, attn = out
            for i in range(len(kps_real)):
                plot_val_kps(
                    kps_real[i],
                    kps_fake[i],
                    pids[i],
                    self.current_epoch,
                    self._data_type,
                )

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    def anomaly_score(self, kps_real, kps_fake, kps_mask, f_real, f_fake):
        B, T = kps_real.size()[:2]

        kps_real = kps_real.view(B, T, self.n_kps, 2)
        kps_fake = kps_fake.view(B, T, self.n_kps, 2)

        # apply mask
        kps_real *= kps_mask
        kps_fake *= kps_mask

        # calc the difference between real keypoints and fake keypoints
        n_nomasked = torch.sum(kps_mask.int().view(B, -1), dim=1)
        loss_residual = torch.abs(kps_real - kps_fake)
        loss_residual = loss_residual.view(B, -1)
        loss_residual = torch.sum(loss_residual, dim=1) / n_nomasked  # mean

        # calc the absolute difference between real feature and fake feature
        loss_discrimination = torch.abs(f_real - f_fake)
        loss_discrimination = loss_discrimination.view(B, -1)
        loss_discrimination = torch.mean(loss_discrimination, dim=1)

        return loss_residual, loss_discrimination

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, kps_real, mask = batch

        # predict
        kps_fake, z, attn, f_real, f_fake = self(kps_real, mask)

        mask_batch_bool = torch.where(mask < 0, False, True)
        l_resi, l_disc = self.anomaly_score(
            kps_real,
            kps_fake,
            mask_batch_bool,
            f_real,
            f_fake,
        )

        # to numpy
        frame_nums = self._to_numpy(frame_nums)
        kps_real = self._to_numpy(kps_real)
        kps_fake = self._to_numpy(kps_fake)
        z = self._to_numpy(z)
        attn = self._to_numpy(attn)
        f_real = self._to_numpy(f_real)
        f_fake = self._to_numpy(f_fake)
        l_resi = self._to_numpy(l_resi)
        l_disc = self._to_numpy(l_disc)

        preds = []
        for frame_num, pid, kr, kf, z_, a, fr, ff, lr, ld in zip(
            frame_nums,
            pids,
            kps_real,
            kps_fake,
            z,
            attn,
            f_real,
            f_fake,
            l_resi,
            l_disc,
        ):
            preds.append(
                {
                    IndividualDataFormat.frame_num: frame_num,
                    IndividualDataFormat.id: pid,
                    IndividualDataFormat.kps_real: kr,
                    IndividualDataFormat.kps_fake: kf,
                    IndividualDataFormat.z: z_,
                    IndividualDataFormat.attn: a,
                    IndividualDataFormat.f_real: fr,
                    IndividualDataFormat.f_fake: ff,
                    IndividualDataFormat.loss_r: lr,
                    IndividualDataFormat.loss_d: ld,
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
            g_optim, [m50, m70, m90], self._config.optim.lr_rate
        )
        d_sdl = torch.optim.lr_scheduler.MultiStepLR(
            d_optim, [m50, m70, m90], self._config.optim.lr_rate
        )

        return [g_optim, d_optim], [g_sdl, d_sdl]
