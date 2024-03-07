import os
from abc import ABCMeta, abstractmethod
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from lightning.pytorch import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modules.utils import json_handler
from modules.utils.constants import Stages


class AbstractDatamodule(LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        data_dir: str,
        config: SimpleNamespace,
        stage: str = Stages.inference,
    ):
        super().__init__()
        self._config = config.dataset
        self._stage = stage

        self._data_dirs = sorted(glob(os.path.join(data_dir, "*/")))

    @property
    def data_dirs(self) -> List[str]:
        return self._data_dirs

    @staticmethod
    def _load_pose_data(
        data_dirs: List[str], data_keys: List[str]
    ) -> List[List[Dict[str, Any]]]:
        pose_data_lst = []
        for pose_data_dir in tqdm(data_dirs, leave=False, ncols=100):
            data = json_handler.load(os.path.join(pose_data_dir, "json", "pose.json"))
            if data is not None:
                pose_data_lst.append(data)
        return pose_data_lst

    @abstractmethod
    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = None,
    ):
        raise NotImplementedError

    def train_dataloader(self, batch_size: int = None):
        assert self._stage == Stages.train
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._train_dataset, batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._val_dataset, batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self, batch_size: int = None):
        return self._test_predict_dataloader(batch_size)

    def predict_dataloader(self, batch_size: int = None):
        return self._test_predict_dataloader(batch_size)

    def _test_predict_dataloader(self, batch_size: int = None):
        assert self._stage is Stages.test or self._stage == Stages.inference
        if batch_size is None:
            batch_size = self._config.batch_size

        dataloaders = [
            DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
            for dataset in self._test_datasets
        ]
        return dataloaders


class AbstractDataset(Dataset):
    def __init__(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
        stage: str,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__()

        self._stage = stage
        self._frame_shape = frame_shape

        self._data: List[Tuple[int, int, NDArray, NDArray]] = []

        self._create_dataset(pose_data, seq_len, th_split)

    def get_data(self, frame_num, pid):
        for data in self._data:
            if frame_num == data[0] and pid == data[1]:
                return data
        return None

    @abstractmethod
    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
    ):
        raise NotImplementedError

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray, NDArray]:
        return self._data[idx]
