import torch
from modules.individual import IndividualDataFormat
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator


class IndividualEGAN(LightningModule):
    def __init__(self, config, data_type):
        super().__init__()

        self._config = config
        self._data_type = data_type
        self._G = Generator(config.model.G, data_type)
        self._D = Discriminator(config.model.D, data_type)
        self._E = Encoder(config.model.E, data_type)
        self._d_z = config.model.G.d_z
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="egan_gloss_{fold}_{epoch}",
                monitor="g_loss",
                mode="max",
            ),
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="egan_dloss_{fold}_{epoch}",
                monitor="d_loss",
                mode="min",
                save_last=True,
            ),
            ModelCheckpoint(
                config.checkpoint_dir,
                filename="egan_eloss_{fold}_{epoch}",
                monitor="e_loss",
                mode="max",
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

    def forward(self, kps_real):
        # predict Z and fake keypoints
        z, _, w_sp, w_tm = self._E(kps_real)
        kps_fake, _ = self._G(z)

        # extract feature maps of real keypoints and fake keypoints from D
        _, f_real = self._D(kps_real, z)
        _, f_fake = self._D(kps_fake, z)

        return z, w_sp, w_tm, kps_fake, f_real, f_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, kps_batch = batch
        batch_size = kps_batch.size()[0]

        # make random noise
        z = torch.randn(batch_size, self._d_z).to(self.device)

        # make true data
        label_real = torch.full((batch_size,), 1, dtype=torch.float32).to(self.device)
        label_fake = torch.full((batch_size,), 0, dtype=torch.float32).to(self.device)

        if optimizer_idx == 0:
            # train Generator
            fake_keypoints, _ = self._G(z)
            d_out_fake, _ = self._D(fake_keypoints, z)

            g_loss = self._criterion(d_out_fake.view(-1), label_real)
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss

        if optimizer_idx == 1:
            # train Discriminator
            z_out_real, _, _, _ = self._E(kps_batch)
            d_out_real, _ = self._D(kps_batch, z_out_real)

            fake_keypoints, _ = self._G(z)
            d_out_fake, _ = self._D(fake_keypoints, z)

            d_loss_real = self._criterion(d_out_real.view(-1), label_real)
            d_loss_fake = self._criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

        if optimizer_idx == 2:
            # train Encoder
            z_out_real, _, _, _ = self._E(kps_batch)
            d_out_real, _ = self._D(kps_batch, z_out_real)

            e_loss = self._criterion(d_out_real.view(-1), label_fake)
            self.log("e_loss", e_loss, prog_bar=True, on_step=True)
            return e_loss

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    @staticmethod
    def anomaly_score(kps_real, kps_fake, f_real, f_fake, lmd):
        kps_real = kps_real.view(kps_real.size()[0], kps_real.size()[1], 17, 2)
        kps_fake = kps_fake.view(kps_fake.size()[0], kps_fake.size()[1], 17, 2)

        # calc the difference between real keypoints and fake keypoints
        loss_residual = torch.norm(kps_real - kps_fake, dim=3)
        loss_residual = loss_residual.view(loss_residual.size()[0], -1)
        loss_residual = torch.mean(loss_residual, dim=1)

        # calc the absolute difference between real feature and fake feature
        loss_discrimination = torch.abs(f_real - f_fake)
        loss_discrimination = loss_discrimination.view(
            loss_discrimination.size()[0], -1
        )
        loss_discrimination = torch.mean(loss_discrimination, dim=1)

        # sum losses
        loss_each = (1 - lmd) * loss_residual + lmd * loss_discrimination

        return loss_each

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, kps_batch = batch
        # predict
        z_lst, w_sp_lst, w_tm_lst, fake_kps_batch, f_real_lst, f_fake_lst = self(
            kps_batch
        )
        anomaly_lst = self.anomaly_score(
            kps_batch, fake_kps_batch, f_real_lst, f_fake_lst, lmd=self._anomaly_lambda
        )

        # to numpy
        frame_nums = self._to_numpy(frame_nums)
        kps_batch = self._to_numpy(kps_batch)
        fake_kps_batch = self._to_numpy(fake_kps_batch)
        z_lst = self._to_numpy(z_lst)
        w_sp_lst = self._to_numpy(w_sp_lst)
        w_tm_lst = self._to_numpy(w_tm_lst)
        anomaly_lst = self._to_numpy(anomaly_lst)

        preds = {}
        for frame_num, pid, z, kps_real, kps_fake, f_real, f_fake, w_sp, w_tm, a in zip(
            frame_nums,
            pids,
            kps_batch,
            fake_kps_batch,
            z_lst,
            w_sp_lst,
            w_tm_lst,
            f_real_lst,
            f_fake_lst,
            anomaly_lst,
        ):
            video_num = pid.split("_")[1]
            if video_num not in preds:
                preds[video_num] = []

            data = {
                IndividualDataFormat.frame_num: frame_num,
                IndividualDataFormat.id: pid,
                IndividualDataFormat.kps_real: kps_real,
                IndividualDataFormat.kps_fake: kps_fake,
                IndividualDataFormat.z: z,
                IndividualDataFormat.w_spat: w_sp,
                IndividualDataFormat.w_temp: w_tm,
                IndividualDataFormat.f_real: f_real,
                IndividualDataFormat.f_fake: f_fake,
                IndividualDataFormat.anomaly: a,
            }
            preds[video_num].append(data)

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
        e_optim = torch.optim.Adam(
            self._E.parameters(),
            self._config.optim.lr_e,
            (self._config.optim.beta1, self._config.optim.beta2),
        )

        return [g_optim, d_optim, e_optim], []
