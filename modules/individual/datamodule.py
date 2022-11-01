import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from modules.pose import PoseDataFormat, PoseDataHandler
from modules.utils.constants import Stages
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .constants import IndividualDataTypes


class IndividualDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        config: SimpleNamespace,
        data_type: str = IndividualDataTypes.local,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
    ):
        if data_type in [IndividualDataTypes.global_, IndividualDataTypes.both]:
            assert frame_shape is not None

        super().__init__()
        self._config = config.dataset
        self._stage = stage

        self._data_dirs = sorted(glob(os.path.join(data_dir, "*")))

        pose_data_lst = self._load_pose_data(self._data_dirs)

        if stage == Stages.train:
            pose_data = []
            for data in pose_data_lst:
                pose_data += data
            self._train_dataset = self._create_dataset(
                pose_data, data_type, frame_shape
            )
            self._val_dataset = IndividualVAlDataset(self._train_dataset)
        elif stage == Stages.test or stage == Stages.inference:
            self._test_datasets = []
            for pose_data in tqdm(pose_data_lst):
                self._test_datasets.append(
                    self._create_dataset(pose_data, data_type, frame_shape)
                )
        else:
            raise NameError

    @property
    def data_dirs(self) -> List[str]:
        return self._data_dirs

    @staticmethod
    def _load_pose_data(data_dirs: List[str]) -> List[List[Dict[str, Any]]]:
        pose_data_lst = []
        for pose_data_dir in data_dirs:
            data = PoseDataHandler.load(
                pose_data_dir, data_keys=[PoseDataFormat.bbox, PoseDataFormat.keypoints]
            )
            if data is not None:
                pose_data_lst.append(data)
        return pose_data_lst

    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        data_type: str = IndividualDataTypes.local,
        frame_shape: Tuple[int, int] = None,
    ):
        return IndividualDataset(
            pose_data,
            self._config.seq_len,
            self._config.th_split,
            self._config.th_mask,
            data_type,
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
        return DataLoader(self._val_dataset, batch_size=batch_size)

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
    # _dmy_kps = np.full((17, 3), np.nan, dtype=np.float32)

    def __init__(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
        th_mask: float,
        data_type: str,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__()

        self._th_mask = th_mask
        self._data_type = data_type
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
        pre_kps = pose_data[0][PoseDataFormat.keypoints]

        seq_data: list = []
        for item in tqdm(pose_data, leave=False):
            # get values
            frame_num = item[PoseDataFormat.frame_num]
            pid = item[PoseDataFormat.id]
            bbox = item[PoseDataFormat.bbox]
            keypoints = item[PoseDataFormat.keypoints]

            if pid != pre_pid:
                if len(seq_data) > seq_len:
                    self._append(seq_data, seq_len)
                # reset seq_data
                del seq_data
                seq_data = []
            else:
                if (
                    1 < frame_num - pre_frame_num
                    and frame_num - pre_frame_num <= th_split
                ):
                    # fill brank with nan
                    seq_data += [
                        (num, pid, pre_bbox, pre_kps)
                        for num in range(pre_frame_num + 1, frame_num)
                    ]
                elif th_split < frame_num - pre_frame_num:
                    if len(seq_data) > seq_len:
                        self._append(seq_data, seq_len)
                    # reset seq_data
                    del seq_data
                    seq_data = []
                else:
                    pass

            # append keypoints to seq_data
            seq_data.append((frame_num, pid, bbox, keypoints))

            # update frame_num and id
            pre_frame_num = frame_num
            pre_pid = pid
            pre_bbox = bbox
            pre_kps = keypoints
            del frame_num, pid, bbox, keypoints
        else:
            self._append(seq_data, seq_len)
            del seq_data

        del pre_frame_num, pre_pid, pre_bbox, pre_kps
        del pose_data

    def _append(self, seq_data, seq_len):
        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            frame_num = seq_data[i + seq_len - 1][0]
            pid = f"{seq_data[i + seq_len - 1][1]}"
            bbox = np.array([item[2] for item in seq_data[i : i + seq_len]])[:, :4]
            kps = np.array([item[3] for item in seq_data[i : i + seq_len]])
            mask = self._create_mask(kps)

            if self._data_type == IndividualDataTypes.global_:
                # global
                kps = self._scaling_keypoints_global(kps, self._frame_shape)
            elif self._data_type == IndividualDataTypes.local:
                # local
                kps = self._scaling_keypoints_local(bbox, kps)
            else:
                # both
                glb_kps = self._scaling_keypoints_global(kps, self._frame_shape)
                lcl_kps = self._scaling_keypoints_local(bbox, kps)
                kps = np.concatenate([glb_kps, lcl_kps], axis=1)
                mask = np.repeat(mask, 2, axis=0).reshape(kps.shape)
                del glb_kps, lcl_kps

            self._data.append((frame_num, pid, kps, mask))
            del frame_num, pid, bbox, kps, mask

    @staticmethod
    def _scaling_keypoints_global(kps, frame_shape):
        glb_kps = kps[:, :, :2] / frame_shape  # 0-1 scalling
        glb_kps = glb_kps.astype(np.float32)
        return glb_kps

    @staticmethod
    def _scaling_keypoints_local(bbox, kps):
        org = bbox[:, :2]
        wh = bbox[:, 2:] - bbox[:, :2]
        lcl_kps = kps[:, :, :2] - np.repeat(org, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps /= np.repeat(wh, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps = lcl_kps.astype(np.float32)
        del org, wh
        return lcl_kps

    def _create_mask(self, kps):
        mask = np.where(
            kps[:, :, 2] < self._th_mask, -1e10, 0.0
        )  # -inf to nan in softmax of attention module
        mask = np.repeat(mask, 2, axis=1).reshape(mask.shape[0], mask.shape[1], 2)
        return mask

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray, NDArray]:
        return self._data[idx]


class IndividualVAlDataset(Dataset):
    val_dataset_settings = [
        (1700, "2"),
        (1700, "9"),
    ]

    def __init__(self, individual_dataset):
        super().__init__()
        self._data = []
        for frame_num, pid in self.val_dataset_settings:
            self._data.append(individual_dataset.get_data(frame_num, pid))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray, NDArray]:
        return self._data[idx]
