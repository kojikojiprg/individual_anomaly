import os
from logging import Logger
from typing import Any, Dict, List

from modules.utils import pickle_handler, video


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
