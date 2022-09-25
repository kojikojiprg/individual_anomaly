import os
from logging import Logger
from typing import Any, Dict, List

import yaml
from modules.utils import pickle_handler, video

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

    def predict(self, video_path: str, data_dir: str = None):
        cap = PoseDataHandler.create_video_capture(video_path, self._logger)

        results = self._model.predict(cap)
        if data_dir is not None:
            PoseDataHandler.save(data_dir, results, self._logger)

        return results


class PoseDataHandler:
    @staticmethod
    def create_video_capture(video_path: str, logger: Logger) -> video.Capture:
        logger.info(f"=> loading video from {video_path}")
        cap = video.Capture(video_path)
        assert cap.is_opened, f"{video_path} does not exist or is wrong file type."
        return cap

    @staticmethod
    def load(data_dir, logger: Logger) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "pose.pkl")

        logger.info(f"=> loading pose estimation results from {pkl_path}")
        pkl_data = pickle_handler.load(pkl_path)
        return pkl_data

    @staticmethod
    def save(data_dir, data: List[dict], logger: Logger):
        pkl_path = os.path.join(data_dir, "pickle", "pose.pkl")

        logger.info(f"=> saving pose estimation results to {pkl_path}")
        pickle_handler.dump(data, pkl_path)
