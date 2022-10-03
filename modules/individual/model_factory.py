import os
from logging import Logger
from types import SimpleNamespace

from .model import IndividualGAN


class IndividualModelFactory:
    @staticmethod
    def create_model(
        config: SimpleNamespace, device: str, logger: Logger
    ) -> IndividualGAN:
        model = IndividualGAN(config, device, logger)
        return model

    @staticmethod
    def load_model(
        checkpoint_dir: str,
        config: SimpleNamespace,
        device: str,
        logger: Logger,
    ) -> IndividualGAN:
        model = IndividualGAN(config, device, logger)
        model.load_checkpoints(checkpoint_dir)
        return model

    @staticmethod
    def save_model(
        model: IndividualGAN,
        checkpoint_dir: str,
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_checkpoints(checkpoint_dir)
