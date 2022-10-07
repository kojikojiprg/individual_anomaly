from logging import Logger
from typing import List

from modules.utils import set_random

from .data_handler import IndividualDataHandler
from .format import Format as IndividualDataFormat
from .model_factory import IndividualModelFactory, IndividualModelType


class IndividualActivityRecognition:
    def __init__(self, model_type: str, device: str, logger: Logger):
        assert model_type.casefold() in IndividualModelType.get_types()

        self._model_type = model_type
        self._config = IndividualDataHandler.get_config(model_type)
        self._device = device
        self._logger = logger
        set_random.seed(self._config.seed)

    def train(self, dirs: List[str], load_model: bool = False):
        # creating dataloader
        dataloader = IndividualDataHandler.create_data_loader(
            dirs, self._config, self._logger
        )

        if load_model:
            # load model
            model = IndividualModelFactory.load_model(
                self._model_type,
                self._config,
                self._device,
                self._logger,
            )
        else:
            # create new model
            model = IndividualModelFactory.create_model(
                self._model_type, self._config, self._device, self._logger
            )

        # train
        model.train(dataloader)

        # save params
        IndividualModelFactory.save_model(model)

    def inference_generator(self, data_dir: str, num_individual: int):
        # load model
        model = IndividualModelFactory.load_model(
            self._model_type, self._config, self._device, self._logger
        )
        results = model.infer_generator(num_individual)
        IndividualDataHandler.save_generator_data(data_dir, results, self._logger)

    def inference_discriminator(self, data_dir: str, num_individual: int):
        # load model
        model = IndividualModelFactory.load_model(
            self._model_type, self._config, self._device, self._logger
        )
        results = model.infer_discriminator(num_individual)
        IndividualDataHandler.save_discriminator_data(data_dir, results, self._logger)
