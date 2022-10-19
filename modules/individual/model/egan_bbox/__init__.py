import torch
from modules.individual import IndividualDataFormat, IndividualDataTypes
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator


class IndividualEganBbox(LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._E = Encoder(config.model.E)
        self._d_z = config.model.G.d_z
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"egan_{IndividualDataTypes.global_bbox}_"
                + "{d_loss:.2f}_{epoch}",
                monitor="d_loss",
                mode="min",
                save_last=True,
            ),
        ]
        self._callbacks[
            0
        ].CHECKPOINT_NAME_LAST = f"egan_last-{IndividualDataTypes.global_bbox}"

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

    def forward(self, bbox_real):
        # predict Z and fake keypoints
        z, w_e = self._E(bbox_real)
        bbox_fake, _ = self._G(z)

        # extract feature maps of real keypoints and fake keypoints from D
        _, f_real = self._D(bbox_real, z)
        _, f_fake = self._D(bbox_fake, z)

        return z, w_e, bbox_fake, f_real, f_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, bbox_batch, _ = batch
        batch_size = bbox_batch.size()[0]

        # make random noise
        z = torch.randn(batch_size, self._d_z).to(self.device)

        # make true data
        label_real = torch.full((batch_size,), 1, dtype=torch.float32).to(self.device)
        label_fake = torch.full((batch_size,), 0, dtype=torch.float32).to(self.device)

        if optimizer_idx == 0:
            # train Generator
            fake_bbox, _ = self._G(z)
            d_out_fake, _ = self._D(fake_bbox, z)

            g_loss = self._criterion(d_out_fake.view(-1), label_real)
            self.log("g_loss", g_loss, prog_bar=True, on_step=True)
            return g_loss

        if optimizer_idx == 1:
            # train Discriminator
            z_out_real, _ = self._E(bbox_batch)
            d_out_real, _ = self._D(bbox_batch, z_out_real)

            fake_bbox, _ = self._G(z)
            d_out_fake, _ = self._D(fake_bbox, z)

            d_loss_real = self._criterion(d_out_real.view(-1), label_real)
            d_loss_fake = self._criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True, on_step=True)
            return d_loss

        if optimizer_idx == 2:
            # train Encoder
            z_out_real, _ = self._E(bbox_batch)
            d_out_real, _ = self._D(bbox_batch, z_out_real)

            e_loss = self._criterion(d_out_real.view(-1), label_fake)
            self.log("e_loss", e_loss, prog_bar=True, on_step=True)
            return e_loss

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    def anomaly_score(self, bbox_real, bbox_fake, f_real, f_fake, lmd):
        bbox_real = bbox_real.view(
            bbox_real.size()[0], bbox_real.size()[1], self.n_kps, 2
        )
        bbox_fake = bbox_fake.view(
            bbox_fake.size()[0], bbox_fake.size()[1], self.n_kps, 2
        )

        # calc the difference between real keypoints and fake keypoints
        loss_residual = torch.norm(bbox_real - bbox_fake, dim=3)
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

        return loss_each, loss_residual, loss_discrimination

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, bbox_batch, _ = batch

        # predict
        z_lst, w_lst, fake_kps_batch, f_real_lst, f_fake_lst = self(bbox_batch)

        anomaly_lst, resi_lst, disc_lst = self.anomaly_score(
            bbox_batch,
            fake_kps_batch,
            f_real_lst,
            f_fake_lst,
            lmd=self._anomaly_lambda,
        )

        # to numpy
        frame_nums = self._to_numpy(frame_nums)
        # kps_batch = self._to_numpy(kps_batch)
        # fake_kps_batch = self._to_numpy(fake_kps_batch)
        # z_lst = self._to_numpy(z_lst)
        # w_lst = self._to_numpy(w_lst)
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
        e_optim = torch.optim.Adam(
            self._E.parameters(),
            self._config.optim.lr_e,
            (self._config.optim.beta1, self._config.optim.beta2),
        )

        return [g_optim, d_optim, e_optim], []
