import torch
from modules.individual import IndividualDataFormat
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .generator import Generator


class IndividualGanomaly(LightningModule):
    def __init__(self, config, data_type):
        super().__init__()

        self._config = config
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._data_type = data_type

        self._G = Generator(config.model.G, data_type)
        self._D = Discriminator(config.model.D, data_type)
        self._l_adv = torch.nn.BCEWithLogitsLoss()
        self._l_con = torch.nn.L1Loss()
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
        z, _, w_sp, w_tm = self._E(kps_real, mask_batch)
        kps_fake, _ = self._G(z)

        # extract feature maps of real keypoints and fake keypoints from D
        _, f_real = self._D(kps_real, z, mask_batch)
        _, f_fake = self._D(kps_fake, z, mask_batch)

        return z, w_sp, w_tm, kps_fake, f_real, f_fake

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
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss

        if optimizer_idx == 1:
            # discriminator loss
            l_adv_real = self._l_adv(pred_real.view(-1), label_real)
            l_adv_fake = self._l_adv(pred_fake.view(-1), label_fake)
            d_loss = l_adv_real + l_adv_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    def anomaly_score(self, kps_real, kps_fake, kps_mask, f_real, f_fake, lmd):
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

        # sum losses
        loss_each = (1 - lmd) * loss_residual + lmd * loss_discrimination

        return loss_each, loss_residual, loss_discrimination

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, kps_batch, mask_batch = batch

        # predict
        z_lst, w_sp_lst, w_tm_lst, fake_kps_batch, f_real_lst, f_fake_lst = self(
            kps_batch, mask_batch
        )

        mask_batch_bool = torch.where(mask_batch < 0, False, True)
        anomaly_lst, resi_lst, disc_lst = self.anomaly_score(
            kps_batch,
            fake_kps_batch,
            mask_batch_bool,
            f_real_lst,
            f_fake_lst,
            lmd=self._anomaly_lambda,
        )

        # to numpy
        frame_nums = self._to_numpy(frame_nums)
        # kps_batch = self._to_numpy(kps_batch)
        # fake_kps_batch = self._to_numpy(fake_kps_batch)
        # z_lst = self._to_numpy(z_lst)
        # w_sp_lst = self._to_numpy(w_sp_lst)
        # w_tm_lst = self._to_numpy(w_tm_lst)
        # f_real_lst = self._to_numpy(f_real_lst)
        # f_fake_lst = self._to_numpy(f_fake_lst)
        anomaly_lst = self._to_numpy(anomaly_lst)
        resi_lst = self._to_numpy(resi_lst)
        disc_lst = self._to_numpy(disc_lst)

        preds = []
        for frame_num, pid, a, r, d in zip(
            frame_nums,
            pids,
            anomaly_lst,
            resi_lst,
            disc_lst,
        ):
            preds.append(
                {
                    IndividualDataFormat.frame_num: frame_num,
                    IndividualDataFormat.id: pid,
                    IndividualDataFormat.anomaly: a,
                    IndividualDataFormat.loss_r: r,
                    IndividualDataFormat.loss_d: d,
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

        return [g_optim, d_optim], []
