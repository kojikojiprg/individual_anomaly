import gc
import os
from typing import List, Tuple, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from modules.utils import set_random
from modules.utils.constants import Stages

from .constants import DataFormat, DataTypes
from .datahandler import DataHandler
from .models.ganomaly_bbox import GanomalyBbox
from .models.ganomaly_bbox.datamodule import DatamoduleBbox
from .models.ganomaly_kps import GanomalyKps
from .models.ganomaly_kps.datamodule import DatamoduleKps


class IndividualAnomalyEstimation:
    def __init__(
        self,
        seq_len: int,
        data_type: str = DataTypes.local,
        stage: str = Stages.inference,
        model_version: int = None,
        masking: bool = False,
    ):
        assert DataTypes.includes(data_type)

        self._seq_len = seq_len
        self._masking = masking
        self._data_type = data_type
        self._stage = stage

        self._config = DataHandler.get_config(seq_len, data_type, stage)
        set_random.seed(self._config.seed)

        self._model: LightningModule
        self._trainer: Trainer

        self._create_model()
        if stage == Stages.inference:
            if model_version is not None:
                model_version = f"-v{model_version}"
            else:
                model_version = ""
            checkpoint_path = os.path.join(
                self._config.checkpoint_dir,
                f"ganomaly_{data_type}_seq{seq_len}_last{model_version}.ckpt",
            )
            self._load_model(checkpoint_path)

    def __del__(self):
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_trainer"):
            del self._trainer
        torch.cuda.empty_cache()
        gc.collect()

    @property
    def model(self) -> LightningModule:
        if hasattr(self, "_model"):
            return self._model
        else:
            raise AttributeError

    def _create_model(self):
        if self._data_type != DataTypes.bbox:
            self._model = GanomalyKps(self._config, self._data_type, self._masking)
        else:
            self._model = GanomalyBbox(self._config, self._data_type)

    def _load_model(self, checkpoint_path: str):
        print(f"=> loading model from {checkpoint_path}")
        self._model = self._model.load_from_checkpoint(
            checkpoint_path,
            config=self._config,
            data_type=self._data_type,
            masking=self._masking,
        )

    def _create_datamodule(
        self,
        data_dir: str,
        frame_shape: Tuple[int, int] = None,
    ) -> Union[DatamoduleBbox, DatamoduleKps]:
        print("=> creating dataset")
        return DataHandler.create_datamodule(
            data_dir,
            self._config,
            self._data_type,
            self._stage,
            frame_shape,
        )

    def _build_trainer(self, data_dir, gpu_ids):
        dataset_dir = os.path.dirname(data_dir)
        log_path = os.path.join(dataset_dir, "logs")

        if hasattr(self._config.train, "accumulate_grad_batches"):
            accumulate_grad_batches = self._config.train.accumulate_grad_batches
        else:
            accumulate_grad_batches = None

        if self._stage == Stages.train:
            logger = TensorBoardLogger(log_path, name=self._data_type)
        else:
            logger = False
        return Trainer(
            logger=logger,
            callbacks=self._model.callbacks,
            max_epochs=self._config.train.epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=gpu_ids,
            accelerator="cuda",
            strategy="ddp",
        )

    def train(self, data_dir: str, gpu_ids: List[int]):
        frame_shape = DataHandler.get_frame_shape(data_dir)
        datamodule = self._create_datamodule(data_dir, frame_shape)

        if not hasattr(self, "_trainer"):
            self._trainer = self._build_trainer(data_dir, gpu_ids)
        self._trainer.fit(self._model, datamodule=datamodule)

        del datamodule
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _collect_results(preds_lst):
        results_lst = []
        for preds in preds_lst:
            results = []
            for batch in preds:
                results += batch
            results_lst.append(results)
        return results_lst

    def inference(
        self,
        data_dir: str,
        gpu_ids: List[int],
    ):
        frame_shape = DataHandler.get_frame_shape(data_dir)
        datamodule = self._create_datamodule(data_dir, frame_shape)

        if not hasattr(self, "_trainer"):
            self._trainer = self._build_trainer(data_dir, gpu_ids)
        preds_lst = self._trainer.predict(
            self._model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=True,
        )
        results_lst = self._collect_results(preds_lst)
        data_dirs = datamodule.data_dirs

        print("=> saving results")
        for path, results in zip(tqdm(data_dirs), results_lst):
            DataHandler.save(
                path,
                results,
                self._data_type,
                self._masking,
                self._seq_len,
            )

        del datamodule
        torch.cuda.empty_cache()
        gc.collect()

        return results_lst
