import os
from typing import Any, Dict, List, Optional, Tuple

from modules.utils import json_handler, video

from .format import Format


class PoseDataHandler:
    @staticmethod
    def create_video_capture(video_path: str) -> video.Capture:
        print(f"=> loading video from {video_path}")
        cap = video.Capture(video_path)
        assert cap.is_opened, f"{video_path} does not exist or is wrong file type."
        return cap

    @staticmethod
    def load(data_dir, data_keys: list = None) -> Optional[List[Dict[str, Any]]]:
        pkl_path = os.path.join(data_dir, "pickle", "pose.pkl")

        data = json_handler.load(pkl_path)
        if data_keys is None:
            return data
        else:
            if data_keys is not None:
                data_keys = data_keys + [Format.frame_num, Format.id]
                data_keys + list(set(data_keys))  # get unique
            else:
                data_keys = [Format.frame_num, Format.id]

            ret_data = [{k: item[k] for k in data_keys} for item in data]
            del data
            return ret_data

    @staticmethod
    def save(data_dir, data: List[dict]):
        pkl_path = os.path.join(data_dir, "json", "pose.pkl")
        json_handler.dump(data, pkl_path)

    @staticmethod
    def save_frame_shape(data_dir, frame_shape: Tuple[int, int]):
        pkl_path = os.path.join(data_dir, "json", "frame_shape.pkl")
        json_handler.dump(frame_shape, pkl_path)
