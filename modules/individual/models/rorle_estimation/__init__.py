from types import SimpleNamespace

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from modules.individual import IndividualDataFormat, IndividualPredTypes

from .model_core import ModelCore


class RoleEstimation(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        data_type: str,
        masking: bool,
        prediction_type: str = None,
    ):
        super().__init__()

        self._config = config
        self.n_kps = config.n_kps
        self._anomaly_lambda = config.inference.anomaly_lambda
        self._data_type = data_type
        self._masking = masking
        self._prediction_type = prediction_type

        self._net = ModelCore(config.model)
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
                filename=f"re{masked_str}_{data_type}_seq{seq_len}_min",
                monitor="re_loss",
                mode="min",
                save_last=True,
            ),
        ]

        last_name = f"re{masked_str}_{data_type}_seq{seq_len}_last"
        self._callbacks[0].CHECKPOINT_NAME_LAST = last_name

    @property
    def callbacks(self) -> list:
        return self._callbacks

    def forward(self, kps, mask):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx):
        pass

    def configure_optimizers(self):
        pass
