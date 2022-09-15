import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from modules.utils import json_handler, video
from numpy.typing import NDArray
from tqdm import tqdm

from .keypoints import KPClass


@dataclass(frozen=True)
class DataFormat:
    frame: str = "frame"
    id: str = "id"
    keypoints: str = "keypoints"


class DataHandler:
    @staticmethod
    def create_video_loader(video_path: str) -> torch.utils.data.DataLoader:
        cap = video.Capture(video_path)
        assert cap.is_opened, f"{video_path} does not exist or is wrong file type."
        dataset = KeypointDetaset(cap)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    @staticmethod
    def _convert(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            DataFormat.frame: item[DataFormat.frame],
            DataFormat.id: item[DataFormat.frame],
            DataFormat.keypoints: KPClass(item[DataFormat.keypoints]),
        }

    @staticmethod
    def load(json_path) -> List[Dict[str, Any]]:
        json_data = json_handler.load(json_path)
        data = [DataHandler._convert(item) for item in tqdm(json_data, ncols=100)]

        return data

    @staticmethod
    def _reform(item: Tuple[int, int, NDArray]):
        return {
            DataFormat.frame: item[0],
            DataFormat.id: item[1],
            DataFormat.keypoints: item[2],
        }

    @staticmethod
    def save(json_path, data: List[Tuple[int, int, NDArray]]):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        reformed_data = []
        for item in data:
            item = DataHandler._reform(item)
            reformed_data.append(item)
        json_handler.dump(json_path, reformed_data)
        del reformed_data


class KeypointDetaset(torch.utils.data.Dataset):
    def __init__(self, cap: video.Capture):
        self._cap = cap
        self._cap.set_pos_frame_count(0)

    def __len__(self):
        return self._cap.frame_count

    def __getitem__(self, idx):
        is_opened, frame = self._cap.read()
        if not is_opened:
            raise FileNotFoundError
        return idx + 1, frame
