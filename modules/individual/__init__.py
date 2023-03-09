import os
from typing import List, Tuple, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from modules.utils import set_random
from modules.utils.constants import Stages

from .constants import (
    IndividualDataFormat,
    IndividualDataTypes,
    IndividualModelTypes,
    IndividualPredTypes,
)
from .datahandler import IndividualDataHandler
from .models.ganomaly_kps import IndividualGanomalyKps
from .models.ganomaly_kps.datamodule import IndividualDataModuleKps
from .models.ganomaly_bbox import IndividualGanomalyBbox
from .models.ganomaly_bbox.datamodule import IndividualDataModuleBbox


class IndividualActivityRecognition:
    def __init__(
        self,
        model_type: str,
        seq_len: int,
        checkpoint_path: str = None,
        data_type: str = IndividualDataTypes.local,
        masking: bool = False,
        stage: str = Stages.inference,
        prediction_type: str = IndividualPredTypes.anomaly,
    ):
        assert IndividualModelTypes.includes(model_type)
        assert IndividualDataTypes.includes(data_type)

        self._model_type = model_type.casefold()
        self._seq_len = seq_len
        self._data_type = data_type
        self._masking = masking
        self._stage = stage
        self._prediction_type = prediction_type

        self._config = IndividualDataHandler.get_config(
            model_type, seq_len, data_type, stage
        )
        set_random.seed(self._config.seed)

        self._model: LightningModule
        self._trainer: Trainer

        self._create_model()
        if checkpoint_path is not None:
            self._load_model(checkpoint_path)

    def __del__(self):
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_trainer"):
            del self._trainer
        torch.cuda.empty_cache()

    @property
    def model(self) -> LightningModule:
        if hasattr(self, "_model"):
            return self._model
        else:
            raise AttributeError

    def _create_model(self):
        if self._data_type != IndividualDataTypes.bbox:
            self._model = IndividualGanomalyKps(
                self._config, self._data_type, self._masking
            )
        else:
            self._model = IndividualGanomalyBbox(self._config, self._data_type)

    def _load_model(self, checkpoint_path: str):
        print(f"=> loading model from {checkpoint_path}")
        self._model = self._model.load_from_checkpoint(
            checkpoint_path,
            config=self._config,
            data_type=self._data_type,
            masking=self._masking,
            prediction_type=self._prediction_type,
        )

    def _create_datamodule(
        self, data_dir: str, frame_shape: Tuple[int, int] = None
    ) -> Union[IndividualDataModuleBbox, IndividualDataModuleKps]:
        print("=> creating dataset")
        return IndividualDataHandler.create_datamodule(
            data_dir, self._config, self._data_type, self._stage, frame_shape
        )

    def _build_trainer(self, data_dir, gpu_ids):
        dataset_dir = os.path.dirname(data_dir)
        log_path = os.path.join(dataset_dir, "logs", self._model_type)

        if hasattr(self._config.train, "accumulate_grad_batches"):
            accumulate_grad_batches = self._config.train.accumulate_grad_batches
        else:
            accumulate_grad_batches = None

        if len(gpu_ids) > 1:
            strategy = "ddp"
        else:
            strategy = None

        if self._stage == Stages.train:
            logger = TensorBoardLogger(log_path, name=self._data_type)
        else:
            logger = False
        return Trainer(
            logger,
            callbacks=self._model.callbacks,
            max_epochs=self._config.train.epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=gpu_ids,
            accelerator="gpu",
            strategy=strategy,
        )

    def train(self, data_dir: str, gpu_ids: List[int]):
        frame_shape = IndividualDataHandler.get_frame_shape(data_dir)
        datamodule = self._create_datamodule(data_dir, frame_shape)

        if not hasattr(self, "_trainer"):
            self._trainer = self._build_trainer(data_dir, gpu_ids)
        self._trainer.fit(self._model, datamodule=datamodule)

        del datamodule
        torch.cuda.empty_cache()

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
        frame_shape = IndividualDataHandler.get_frame_shape(data_dir)
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
            IndividualDataHandler.save(
                path,
                results,
                self._model_type,
                self._data_type,
                self._masking,
                self._seq_len,
                self._prediction_type,
            )

        del datamodule
        torch.cuda.empty_cache()

        return results_lst
