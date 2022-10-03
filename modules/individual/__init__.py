import os
from logging import Logger
from typing import List

from .data_handler import IndividualDataHandler
from .format import Format as IndividualDataFormat
from .model import IndividualGAN
from .model_factory import IndividualModelFactory


class IndividualActivityRecognition:
    def __init__(self, config_path: str, device: str, logger: Logger):
        self._config = IndividualDataHandler.get_config(config_path)
        self._device = device
        self._logger = logger

    def train(self, dirs: List[str], checkpoint_dir: str, load_model: bool = False):
        # creating dataloader
        dataloader = IndividualDataHandler.create_data_loader(
            dirs, self._config, self._logger
        )

        if load_model and os.path.exists(checkpoint_dir):
            # load model
            model = IndividualModelFactory.load_model(
                checkpoint_dir, self._config, self._device, self._logger
            )
        else:
            # create new model
            model = IndividualModelFactory.create_model(
                self._config, self._device, self._logger
            )

        # train
        model.train(dataloader)

        # save params
        IndividualModelFactory.save_model(model, checkpoint_dir)

    def inference(self, checkpoint_dir: str, data_dir: str, num_individual: int):
        # load model
        model = IndividualModelFactory.load_model(
            checkpoint_dir, self._config, self._device, self._logger
        )
        results = model.test_generator(num_individual)
        IndividualDataHandler.save_generator_data(data_dir, results, self._logger)
