import os
from logging import Logger
from typing import Any, Dict, List, Union

from modules.utils import pickle_handler, video

from .format import Format


class PoseDataHandler:
    @staticmethod
    def create_video_capture(video_path: str, logger: Logger) -> video.Capture:
        logger.info(f"=> loading video from {video_path}")
        cap = video.Capture(video_path)
        assert cap.is_opened, f"{video_path} does not exist or is wrong file type."
        return cap

    @staticmethod
    def load(data_dir, data_keys: list = None) -> Union[List[Dict[str, Any]], None]:
        pkl_path = os.path.join(data_dir, "pickle", "pose.pkl")

        data = pickle_handler.load(pkl_path)
        if data_keys is None:
            return data
        else:
            data_keys = set(data_keys + [Format.frame_num, Format.id])
            ret_data = [{k: item[k] for k in data_keys} for item in data]
            del data
            return ret_data

    @staticmethod
    def save(data_dir, data: List[dict]):
        pkl_path = os.path.join(data_dir, "pickle", "pose.pkl")
        pickle_handler.dump(data, pkl_path)
