import os
from dataclasses import dataclass
from logging import Logger
from types import SimpleNamespace
from typing import List, Union

from .model.egan import IndividualEGAN
from .model.gan import IndividualGAN


@dataclass(frozen=True)
class IndividualModelType:
    gan: str = "gan"
    egan: str = "egan"
    autoencoder: str = "autoencoder"

    @classmethod
    def get_types(cls) -> List[str]:
        return list(cls.__dict__.keys())


class IndividualModelFactory:
    @staticmethod
    def create_model(
        model_type: str, config: SimpleNamespace, device: str, logger: Logger
    ) -> IndividualGAN:
        model_type = model_type.lower()

        if model_type == IndividualModelType.gan:
            model = IndividualGAN(config, device, logger)
        elif model_type == IndividualModelType.egan:
            model = IndividualEGAN(config, device, logger)
        elif model_type == IndividualModelType.autoencoder:
            raise NotImplementedError
        else:
            raise NameError

        return model

    @staticmethod
    def load_model(
        model_type: str,
        checkpoint_dir: str,
        config: SimpleNamespace,
        device: str,
        logger: Logger,
    ) -> Union[IndividualGAN, IndividualEGAN]:
        model = IndividualModelFactory.create_model(model_type, config, device, logger)
        model.load_checkpoints(checkpoint_dir)
        return model

    @staticmethod
    def save_model(
        model: Union[IndividualGAN, IndividualEGAN],
        checkpoint_dir: str,
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_checkpoints(checkpoint_dir)
