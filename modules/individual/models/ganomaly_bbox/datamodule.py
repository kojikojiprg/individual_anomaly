import os
import gc
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule
from scipy import interpolate
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from modules.pose import PoseDataFormat, PoseDataHandler
from modules.utils.constants import Stages


class IndividualDataModuleBbox(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        config: SimpleNamespace,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__()
        self._config = config.dataset
        self._stage = stage

        self._data_dirs = sorted(glob(os.path.join(data_dir, "*")))

        pose_data_lst = self._load_pose_data(self._data_dirs)

        if stage == Stages.train:
            pose_data = []
            for data in pose_data_lst:
                pose_data += data
            self._train_dataset = self._create_dataset(pose_data, frame_shape)
            val_ids = np.random.randint(
                len(self._train_dataset), size=self._config.n_val
            )
            self._val_dataset = Subset(self._train_dataset, val_ids)
        elif stage == Stages.test or stage == Stages.inference:
            self._test_datasets = []
            for pose_data in tqdm(pose_data_lst):
                self._test_datasets.append(self._create_dataset(pose_data, frame_shape))
        else:
            raise NameError

    @property
    def data_dirs(self) -> List[str]:
        return self._data_dirs

    @staticmethod
    def _load_pose_data(data_dirs: List[str]) -> List[List[Dict[str, Any]]]:
        pose_data_lst = []
        for pose_data_dir in data_dirs:
            data = PoseDataHandler.load(pose_data_dir, data_keys=[PoseDataFormat.bbox])
            if data is not None:
                pose_data_lst.append(data)
        return pose_data_lst

    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = None,
    ):
        return IndividualDataset(
            pose_data,
            self._config.seq_len,
            self._config.th_split,
            self._stage,
            frame_shape,
        )

    def train_dataloader(self, batch_size: int = None):
        assert self._stage == Stages.train
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._train_dataset, batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._val_dataset, batch_size, shuffle=False, num_workers=8)

    def _test_predict_dataloader(self, batch_size: int = None):
        assert self._stage is Stages.test or self._stage == Stages.inference
        if batch_size is None:
            batch_size = self._config.batch_size

        dataloaders = [
            DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
            for dataset in self._test_datasets
        ]
        return dataloaders

    def test_dataloader(self, batch_size: int = None):
        return self._test_predict_dataloader(batch_size)

    def predict_dataloader(self, batch_size: int = None):
        return self._test_predict_dataloader(batch_size)


class IndividualDataset(Dataset):
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

    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
    ):
        # sort data by frame_num
        pose_data = sorted(pose_data, key=lambda x: x[PoseDataFormat.frame_num])
        # sort data by id
        pose_data = sorted(pose_data, key=lambda x: x[PoseDataFormat.id])

        # get frame_num and id of first data
        pre_frame_num = pose_data[0][PoseDataFormat.frame_num]
        pre_pid = pose_data[0][PoseDataFormat.id]
        pre_bbox = pose_data[0][PoseDataFormat.bbox]

        seq_data: list = []
        for item in tqdm(pose_data, leave=False):
            # get values
            frame_num = item[PoseDataFormat.frame_num]
            pid = item[PoseDataFormat.id]
            bbox = item[PoseDataFormat.bbox]

            if pid != pre_pid:
                if len(seq_data) > seq_len:
                    self._append(seq_data, seq_len)
                # reset seq_data
                seq_data = []
            else:
                if (
                    1 < frame_num - pre_frame_num
                    and frame_num - pre_frame_num <= th_split
                ):
                    # fill brank with nan
                    seq_data += [
                        (num, pid, np.full((4,), np.nan))
                        for num in range(pre_frame_num + 1, frame_num)
                    ]
                elif th_split < frame_num - pre_frame_num:
                    if len(seq_data) > seq_len:
                        self._append(seq_data, seq_len)
                    # reset seq_data
                    seq_data = []
                else:
                    pass

            # replace nan into low confidence points
            # bbox = np.full((4,), np.nan) if bbox[4] < 0.2 else bbox

            # append keypoints to seq_data
            seq_data.append((frame_num, pid, bbox))

            # update frame_num and id
            pre_frame_num = frame_num
            pre_pid = pid
            pre_bbox = bbox
        else:
            if len(seq_data) > seq_len:
                self._append(seq_data, seq_len)

        del pre_frame_num, pre_pid, pre_bbox
        del pose_data, seq_data
        gc.collect()

    def _append(self, seq_data, seq_len):
        # collect bbox
        all_bboxs = np.array([item[2][:4] for item in seq_data])

        # delete last nan
        # print(len(all_bboxs), all_bboxs[-1, -1])
        # all_bboxs = all_bboxs[:max(np.where(~np.isnan(all_bboxs))[0]) + 1]
        # print(len(all_bboxs), all_bboxs[-1, -1])

        # interpolate bbox and keypoints
        all_bboxs = self._interpolate2d(all_bboxs)

        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            frame_num = seq_data[i + seq_len - 1][0]
            pid = f"{seq_data[i + seq_len - 1][1]}"
            bbox = all_bboxs[i : i + seq_len, :4]

            # bbox
            bbox = bbox.reshape(-1, 2, 2)
            bbox = bbox / self._frame_shape  # scalling bbox top-left
            bbox = bbox.astype(np.float32)
            self._data.append((frame_num, pid, bbox))

    @staticmethod
    def _interpolate2d(vals: NDArray):
        ret_vals = np.empty((vals.shape[0], 0))
        for i in range(vals.shape[1]):
            x = np.where(~np.isnan(vals[:, i]))[0]
            y = vals[:, i][~np.isnan(vals[:, i])]
            fitted_curve = interpolate.interp1d(x, y, kind="cubic")
            fitted_y = fitted_curve(np.arange(len(vals)))
            fitted_y[np.isnan(vals[:, i])] += np.random.normal(
                0, 0.1, len(fitted_y[np.isnan(vals[:, i])])
            )

            ret_vals = np.append(ret_vals, fitted_y.reshape(-1, 1), axis=1)
        return ret_vals

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray]:
        return self._data[idx]
