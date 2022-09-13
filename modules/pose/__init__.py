from logging import Logger
from typing import List, Tuple

import yaml
from numpy.typing import NDArray

from .data_handler import DataFormat, DataHandler
from .keypoints import KPClass, KPListClass, KPName
from .model import KPModel


class PoseEstimation:
    def __init__(
        self,
        cfg_path: str,
        device: str,
        logger: Logger,
    ):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self._model = KPModel(cfg, device, logger)

        self._results: List[Tuple[int, int, NDArray]] = []

    def __del__(self):
        del self._data_loader, self._model

    def predict(self, video_path: str):
        data_loader = DataHandler.create_video_loader(video_path)
        self._results = self._model.predict(data_loader)

    def save_result(self, output_path: str):
        DataHandler.save(output_path, self._results)
