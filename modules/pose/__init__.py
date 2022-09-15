from logging import Logger

import yaml

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

    def __del__(self):
        del self._model

    def predict(self, video_path: str, output_path: str = None):
        data_loader = DataHandler.create_video_loader(video_path)
        results = self._model.predict(data_loader)
        if output_path is not None:
            DataHandler.save(output_path, results)

        return results
