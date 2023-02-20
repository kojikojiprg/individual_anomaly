import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from modules.individual import IndividualDataFormat, IndividualDataTypes

from .discriminator import Discriminator
from .generator import Generator

# from modules.visualize.individual import plot_val_kps


class IndividualGanomalyBbox(LightningModule):
    def __init__(self, config, data_type):
        assert data_type == IndividualDataTypes.bbox
        super().__init__()

        self._config = config
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._data_type = data_type

        self._G = Generator(config.model.G)
        self._D = Discriminator(config.model.D)
        self._l_adv = torch.nn.BCEWithLogitsLoss()
        self._l_con = torch.nn.MSELoss()
        self._l_lat = torch.nn.MSELoss()

        seq_len = config.dataset.seq_len
        self._callbacks = [
            ModelCheckpoint(
                config.checkpoint_dir,
                filename=f"ganomaly_{data_type}_seq{seq_len}_gcon_min",
                monitor="g_l_con",
                mode="min",
                save_last=True,
            ),
        ]
        last_name = f"ganomaly_{data_type}_seq{seq_len}_last"
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

    def forward(self, bbox_real):
        # pred real
        pred_real, f_real = self._D(bbox_real)

        # pred fake
        bbox_fake, z, attn_g = self._G(bbox_real)
        pred_fake, f_fake = self._D(bbox_fake)

        return pred_real, pred_fake, bbox_fake, z, attn_g, f_real, f_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        frame_nums, pids, bbox_real = batch
        batch_size = bbox_real.size()[0]

        # make true data
        label_real = torch.ones((batch_size,), dtype=torch.float32).to(self.device)
        label_fake = torch.zeros((batch_size,), dtype=torch.float32).to(self.device)

        # predict
        pred_real, pred_fake, bbox_fake, _, _, f_real, f_fake = self(bbox_real)

        if optimizer_idx == 0:
            # generator loss
            l_adv = (
                self._l_adv(pred_fake.view(-1), label_real) * self._config.loss.G.w_adv
            )
            l_con = self._l_con(bbox_fake, bbox_real) * self._config.loss.G.w_con
            l_lat = self._l_lat(f_fake, f_real) * self._config.loss.G.w_lat
            g_loss = l_adv + l_con + l_lat
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
        frame_nums, pids, bbox_real, _ = batch
        batch_size = bbox_real.size()[0]

        # make true data
        label_real = np.ones((batch_size,), dtype=np.float32)
        label_fake = np.zeros((batch_size,), dtype=np.float32)

        # predict
        pred_real, pred_fake, bbox_fake, z, attn, _, _ = self(bbox_real)

        bbox_real = bbox_real.cpu().numpy()
        bbox_fake = bbox_fake.cpu().numpy()
        z = z.cpu().numpy()
        attn = attn.cpu().numpy()
        pred_real = pred_real.view(-1).cpu().numpy()
        pred_fake = pred_fake.view(-1).cpu().numpy()

        label = np.concatenate([label_real, label_fake])
        pred = np.concatenate([pred_real, pred_fake])
        d_auc = roc_auc_score(label, pred)
        self.log("d_auc", d_auc)

        # return pids, bbox_real, bbox_fake, z, attn

    # def validation_epoch_end(self, outputs):
    #     for out in outputs:
    #         pids, bbox_real, bbox_fake, z, attn = out
    #         for i in range(len(bbox_real)):
    #             plot_val_kps(
    #                 bbox_real[i],
    #                 bbox_fake[i],
    #                 pids[i],
    #                 self.current_epoch,
    #                 self._data_type,
    #             )

    @staticmethod
    def _to_numpy(tensor):
        return tensor.cpu().numpy()

    def anomaly_score(self, bbox_real, bbox_fake, f_real, f_fake):
        B, T = bbox_real.size()[:2]

        bbox_real = bbox_real.view(B, T, 2, 2)
        bbox_fake = bbox_fake.view(B, T, 2, 2)

        # calc the difference between real keypoints and fake keypoints
        loss_residual = torch.abs(bbox_real - bbox_fake)
        loss_residual = loss_residual.view(B, -1)
        loss_residual = torch.mean(loss_residual, dim=1)

        # calc the absolute difference between real feature and fake feature
        loss_discrimination = torch.abs(f_real - f_fake)
        loss_discrimination = loss_discrimination.view(B, -1)
        loss_discrimination = torch.mean(loss_discrimination, dim=1)

        return loss_residual, loss_discrimination

    def predict_step(self, batch, batch_idx, dataloader_idx):
        frame_nums, pids, bbox_real, _ = batch

        # predict
        _, _, bbox_fake, z, attn, f_real, f_fake = self(bbox_real)

        l_resi, l_disc = self.anomaly_score(
            bbox_real,
            bbox_fake,
            f_real,
            f_fake,
        )

        # to numpy
        frame_nums = self._to_numpy(frame_nums)
        bbox_real = self._to_numpy(bbox_real)
        bbox_fake = self._to_numpy(bbox_fake)
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
            bbox_real,
            bbox_fake,
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
            g_optim, [m50, m70, m90], self._config.optim.lr_rate_g
        )
        d_sdl = torch.optim.lr_scheduler.MultiStepLR(
            d_optim, [m50, m70, m90], self._config.optim.lr_rate_d
        )

        return [g_optim, d_optim], [g_sdl, d_sdl]
