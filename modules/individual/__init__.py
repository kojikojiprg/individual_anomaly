import os
from logging import Logger
from typing import List

from modules.utils import set_random
from modules.utils.constants import Stages
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from .constants import IndividualDataFormat, IndividualDataTypes, IndividualModelTypes
from .datahandler import IndividualDataHandler
from .datamodule import IndividualDataModule
from .model.autoencoder import IndividualAutoencoder
from .model.egan import IndividualEGAN
from .model.gan import IndividualGAN


class IndividualActivityRecognition:
    def __init__(
        self,
        model_type: str,
        logger: Logger,
        checkpoint_path: str = None,
        data_type: str = IndividualDataTypes.both,
        stage: str = Stages.inference,
    ):
        assert IndividualModelTypes.includes(model_type)
        assert IndividualDataTypes.includes(data_type)

        self._model_type = model_type.casefold()
        self._logger = logger
        self._data_type = data_type
        self._stage = stage

        self._config = IndividualDataHandler.get_config(model_type, stage)
        set_random.seed(self._config.seed)

        self._create_model()
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

    @property
    def model(self) -> LightningModule:
        return self._model

    def _create_model(self):
        if self._model_type == IndividualModelTypes.gan:
            self._model = IndividualGAN(self._config, self._data_type)
        elif self._model_type == IndividualModelTypes.egan:
            self._model = IndividualEGAN(self._config, self._data_type)
        elif self._model_type == IndividualModelTypes.autoencoder:
            self._model = IndividualAutoencoder(self._config, self._data_type)
        else:
            raise NameError

    def load_model(self, checkpoint_path: str) -> LightningModule:
        self._logger.info("=> loading model")
        self._model.load_from_checkpoint(checkpoint_path, config=self._config)
        return self._model

    def _create_datamodule(self, data_dir: str) -> IndividualDataModule:
        self._logger.info("=> creating dataset")
        return IndividualDataHandler.create_datamodule(
            data_dir, self._config, self._data_type, self._stage
        )

    def _build_trainer(self, data_dir, gpu_ids):
        log_path = os.path.join(data_dir, "logs")
        if len(gpu_ids) > 1:
            strategy = "ddp"
        else:
            strategy = None
        return Trainer(
            TensorBoardLogger(log_path, name=self._model_type),
            callbacks=self._model.callbacks,
            max_epochs=self._config.train.epochs,
            devices=gpu_ids,
            accelerator="gpu",
            strategy=strategy,
        )

    def train(self, data_dir: str, gpu_ids: List[int]):
        datamodule = self._create_datamodule(data_dir)
        trainer = self._build_trainer(data_dir, gpu_ids)
        trainer.fit(self._model, datamodule=datamodule)

    def inference(self, data_dir: str, gpu_ids: List[int]):
        datamodule = self._create_datamodule(data_dir)
        trainer = self._build_trainer(data_dir, gpu_ids)
        preds_lst = trainer.predict(
            self._model,
            dataloaders=datamodule.predict_dataloader(),
            return_predictions=True,
        )
        data_dirs = datamodule.data_dirs

        self._logger.info("=> saving results")
        for path, preds in tqdm(zip(data_dirs, preds_lst)):
            IndividualDataHandler.save(path, preds, self._model_type, self._data_type)

        return preds_lst
