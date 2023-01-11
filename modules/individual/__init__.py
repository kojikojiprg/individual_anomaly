import os
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from modules.utils import set_random
from modules.utils.constants import Stages

from .constants import IndividualDataFormat, IndividualDataTypes, IndividualModelTypes
from .datahandler import IndividualDataHandler
from .datamodule import IndividualDataModule
from .models.ganomaly import IndividualGanomaly
from .models.ganomaly_bbox import IndividualGanomalyBbox


class IndividualActivityRecognition:
    def __init__(
        self,
        model_type: str,
        seq_len: int,
        checkpoint_path: str = None,
        data_type: str = IndividualDataTypes.local,
        stage: str = Stages.inference,
    ):
        assert IndividualModelTypes.includes(model_type)
        assert IndividualDataTypes.includes(data_type)

        self._model_type = model_type.casefold()
        self._seq_len = seq_len
        self._data_type = data_type
        self._stage = stage

        self._config = IndividualDataHandler.get_config(
            model_type, data_type, stage, seq_len
        )
        set_random.seed(self._config.seed)

        self._model: LightningModule
        self._trainer: Trainer

        self._create_model()
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

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
            self._model = IndividualGanomaly(self._config, self._data_type)
        else:
            self._model = IndividualGanomalyBbox(self._config, self._data_type)

    def load_model(self, checkpoint_path: str) -> LightningModule:
        print(f"=> loading model from {checkpoint_path}")
        self._model = self._model.load_from_checkpoint(
            checkpoint_path, config=self._config, data_type=self._data_type
        )
        return self._model

    def _create_datamodule(
        self, data_dir: str, frame_shape: Tuple[int, int] = None
    ) -> IndividualDataModule:
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

        return Trainer(
            TensorBoardLogger(log_path, name=self._data_type),
            callbacks=self._model.callbacks,
            max_epochs=self._config.train.epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=gpu_ids,
            accelerator="gpu",
            strategy=strategy,
        )

    def train(self, data_dir: str, gpu_ids: List[int], video_dir: str = None):
        frame_shape = None
        if video_dir is not None:
            frame_shape = IndividualDataHandler.get_frame_shape(video_dir)

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

    def inference(self, data_dir: str, gpu_ids: List[int], video_dir: str = None):
        frame_shape = None
        if video_dir is not None:
            frame_shape = IndividualDataHandler.get_frame_shape(video_dir)

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
                path, results, self._model_type, self._data_type, self._seq_len
            )

        del datamodule
        torch.cuda.empty_cache()

        return results_lst
