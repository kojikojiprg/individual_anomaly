import os
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List, Tuple

from modules.utils import json_handler, video
from numpy.typing import NDArray

from .keypoints import KPClass


@dataclass(frozen=True)
class DataFormat:
    frame: str = "frame"
    id: str = "id"
    keypoints: str = "keypoints"


class DataHandler:
    @staticmethod
    def create_video_capture(video_path: str, logger: Logger) -> video.Capture:
        logger.info(f"=> loading video from {video_path}")
        cap = video.Capture(video_path)
        assert cap.is_opened, f"{video_path} does not exist or is wrong file type."
        return cap

    @staticmethod
    def _convert(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            DataFormat.frame: item[DataFormat.frame],
            DataFormat.id: item[DataFormat.id],
            DataFormat.keypoints: KPClass(item[DataFormat.keypoints]),
        }

    @staticmethod
    def load(data_dir, logger: Logger) -> List[Dict[str, Any]]:
        json_path = os.path.join(data_dir, "json", "pose.json")
        logger.info(f"=> loading pose estimation results from {json_path}")
        json_data = json_handler.load(json_path)
        data = [DataHandler._convert(item) for item in json_data]

        return data

    @staticmethod
    def _reform(item: Tuple[int, int, NDArray]):
        return {
            DataFormat.frame: item[0],
            DataFormat.id: item[1],
            DataFormat.keypoints: item[2],
        }

    @staticmethod
    def save(data_dir, data: List[Tuple[int, int, NDArray]], logger: Logger):
        json_path = os.path.join(data_dir, "json", "pose.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        logger.info(f"=> saving pose estimation results to {json_path}")
        reformed_data = []
        for item in data:
            item = DataHandler._reform(item)
            reformed_data.append(item)
        json_handler.dump(json_path, reformed_data)

        del reformed_data
