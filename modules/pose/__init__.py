from logging import Logger

import yaml

from .data_handler import PoseDataHandler
from .format import Format as PoseDataFormat
from .model import PoseModel


class PoseEstimation:
    def __init__(
        self,
        cfg_path: str,
        device: str,
        logger: Logger,
    ):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self._model = PoseModel(cfg, device, logger)
        self._logger = logger

    def __del__(self):
        del self._model

    def inference(self, video_path: str, data_dir: str = None):
        cap = PoseDataHandler.create_video_capture(video_path, self._logger)

        results = self._model.predict(cap)
        if data_dir is not None:
            self._logger.info(f"=> saving pose estimation results to {data_dir}")
            PoseDataHandler.save(data_dir, results)

        return results
