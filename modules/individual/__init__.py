import os
from dataclasses import dataclass
from logging import Logger
from typing import List

from modules.utils import set_random
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from .datahandler import IndividualDataHandler
from .model.autoencoder import IndividualAutoencoder
from .model.egan import IndividualEGAN
from .model.gan import IndividualGAN


@dataclass(frozen=True)
class IndividualDataFormat:
    frame_num: str = "frame"
    id: str = "id"
    keypoints: str = "keypoints"
    feature: str = "features"
    w_spat: str = "weights_spatial"
    w_temp: str = "weights_temporal"


@dataclass(frozen=True)
class IndividualModelTypes:
    gan: str = "gan"
    egan: str = "egan"
    autoencoder: str = "autoencoder"

    @classmethod
    def get_types(cls) -> List[str]:
        return list(cls.__dict__.keys())

    @classmethod
    def includes(cls, model_type: str):
        return model_type.casefold() in cls.get_types()


class IndividualActivityRecognition:
    def __init__(
        self,
        model_type: str,
        data_dir: str,
        gpu_ids: list,
        logger: Logger,
        stage: str = None,
    ):
        assert IndividualModelTypes.includes(model_type)

        self._model_type = model_type.casefold()
        self._gpu_ids = gpu_ids
        self._logger = logger
        self._stage = stage
        self._config = IndividualDataHandler.get_config(model_type)
        set_random.seed(self._config.seed)

        self._logger.info("=> creating dataset")
        self._datamodule = IndividualDataHandler.create_datamodule(
            data_dir, self._config, stage
        )
        self._logger.info("=> creating model")
        self._create_model()

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

    def train(self, is_continue: bool = False):
        if is_continue:
            ckpt_path = os.path.join(self._config.checkpoint_dir, "")
            self._logger.info(f"=> loading model parameters from {ckpt_path}")
            self._model.load_from_checkpoint()

        # train
        trainer = Trainer(
            strategy="ddp", gpus=self._gpu_ids, callbacks=self._model.callbacks
        )
        trainer.fit(self._model, datamodule=self._datamodule)

    # def inference_generator(self, data_dir: str, num_individual: int):
    #     # load model
    #     model = IndividualModelFactory.load_model(
    #         self._model_type, self._config, self._device, self._logger
    #     )
    #     results = model.infer_generator(num_individual)
    #     IndividualDataHandler.save_generator_data(data_dir, results, self._logger)

    # def inference_discriminator(self, data_dir: str, num_individual: int):
    #     # load model
    #     model = IndividualModelFactory.load_model(
    #         self._model_type, self._config, self._device, self._logger
    #     )
    #     results = model.infer_discriminator(num_individual)
    #     IndividualDataHandler.save_discriminator_data(data_dir, results, self._logger)
