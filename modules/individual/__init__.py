import os
from logging import Logger

from modules.utils import set_random
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from .constants import IndividualDataFormat, IndividualDataTypes, IndividualModelTypes
from .datahandler import IndividualDataHandler
from .model.autoencoder import IndividualAutoencoder
from .model.egan import IndividualEGAN
from .model.gan import IndividualGAN


class IndividualActivityRecognition:
    def __init__(
        self,
        model_type: str,
        data_dir: str,
        gpu_ids: list,
        logger: Logger,
        checkpoint_path: str = None,
        data_type: str = "both",
        stage: str = None,
    ):
        assert IndividualModelTypes.includes(model_type)

        self._model_type = model_type.casefold()
        self._datadir = data_dir
        self._logger = logger
        self._data_type = data_type

        self._config = IndividualDataHandler.get_config(model_type, stage)
        set_random.seed(self._config.seed)

        self._logger.info("=> creating dataset")
        self._datamodule = IndividualDataHandler.create_datamodule(
            data_dir, self._config.dataset, data_type, stage
        )

        self._create_model()
        if checkpoint_path is not None:
            self._logger.info("=> loading model")
            self.load_model(checkpoint_path)

        log_path = os.path.join(data_dir, "logs")
        if len(gpu_ids) > 1:
            strategy = "ddp"
        else:
            strategy = None
        self._trainer = Trainer(
            TensorBoardLogger(log_path, name=self._model_type),
            callbacks=self._model.callbacks,
            max_epochs=self._config.train.epochs,
            devices=gpu_ids,
            accelerator="gpu",
            strategy=strategy,
        )

    @property
    def model(self) -> LightningModule:
        return self._model

    @property
    def datamodule(self) -> LightningDataModule:
        return self._datamodule

    def _create_model(self):
        if self._model_type == IndividualModelTypes.gan:
            self._model = IndividualGAN(self._config)
        elif self._model_type == IndividualModelTypes.egan:
            self._model = IndividualEGAN(self._config)
        elif self._model_type == IndividualModelTypes.autoencoder:
            self._model = IndividualAutoencoder(self._config)
        else:
            raise NameError

    def load_model(self, checkpoint_path: str) -> LightningModule:
        self._model.load_from_checkpoint(checkpoint_path, config=self._config)
        return self._model

    def train(self):
        self._trainer.fit(self._model, datamodule=self._datamodule)

    @staticmethod
    def collect_result(preds):
        results = {}
        for pred in preds:
            for key, vals in pred.items():
                if key not in results:
                    results[key] = []
                results[key] += vals
        return results

    def inference(self):
        train_preds, test_preds = self._trainer.predict(
            self._model,
            dataloaders=self._datamodule.predict_dataloader(),
            return_predictions=True,
        )

        train_results = self.collect_result(train_preds)
        test_results = self.collect_result(test_preds)

        self._logger.info("=> saving results")
        IndividualDataHandler.save_data(
            self._model_type,
            self._datadir,
            {"train": train_results, "test": test_results},
        )

        return train_results, test_results
