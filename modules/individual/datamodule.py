import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from modules.pose import PoseDataFormat, PoseDataHandler
from modules.utils import video
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
    ):
        super().__init__()
        self._config = config.dataset
        self._stage = stage

        self._data_dirs = sorted(glob(os.path.join(data_dir, "*")))

        pose_data_lst = self._load_pose_data(self._data_dirs)
        frame_shape = self._get_frame_shape(self._data_dirs)

        self._datasets = []
        if stage == Stages.train:
            pose_data = []
            for data in pose_data_lst:
                pose_data += data
            self._datasets.append(
                self._create_dataset(pose_data, frame_shape, data_type)
            )
        elif stage == Stages.test or stage == Stages.inference:
            for pose_data in tqdm(pose_data_lst):
                self._datasets.append(
                    self._create_dataset(pose_data, frame_shape, data_type)
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
            data = PoseDataHandler.load(pose_data_dir)
            if data is not None:
                pose_data_lst.append(data)
        return pose_data_lst

    @staticmethod
    def _get_frame_shape(data_dirs: List[str]):
        video_path = os.path.join(data_dirs[0], f"{os.path.basename(data_dirs[0])}.mp4")
        cap = video.Capture(video_path)
        frame_shape = cap.size
        del cap
        return frame_shape

    def _create_dataset(
        self,
        pose_data: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        data_type: str = IndividualDataTypes.local,
    ):
        return IndividualDataset(
            pose_data,
            self._config.seq_len,
            self._config.th_split,
            self._config.th_mask,
            frame_shape,
            data_type,
        )

    def train_dataloader(self, batch_size: int = None):
        assert self._stage == Stages.train
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._datasets[0], batch_size, shuffle=True, num_workers=8)

    def _test_predict_dataloader(self, batch_size: int = None):
        assert self._stage is Stages.test or self._stage == Stages.inference
        if batch_size is None:
            batch_size = self._config.batch_size

        dataloaders = [
            DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
            for dataset in self._datasets
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
        frame_shape_xy: Tuple[int, int],
        data_type: str,
    ):
        super().__init__()

        self._frame_shape_xy = frame_shape_xy
        self._th_mask = th_mask
        self._data_type = data_type

        self._data: List[Tuple[int, int, NDArray, NDArray]] = []

        self._create_dataset(pose_data, seq_len, th_split)

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
        else:
            self._append(seq_data, seq_len)

    def _append(self, seq_data, seq_len):
        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            self._data.append(
                # (frame_num, pid, bbox, keypoints)
                (
                    seq_data[i + seq_len - 1][0],
                    f"{seq_data[i + seq_len - 1][1]}",
                    np.array([item[2] for item in seq_data[i : i + seq_len]])[:, :4],
                    np.array([item[3] for item in seq_data[i : i + seq_len]]),
                )
            )

    def __len__(self):
        return len(self._data)

    @staticmethod
    def _scaling_keypoints_global(kps, frame_shape_xy):
        glb_kps = kps[:, :, :2] / frame_shape_xy  # 0-1 scalling
        glb_kps = glb_kps.astype(np.float32)
        return glb_kps

    @staticmethod
    def _scaling_keypoints_local(bbox, kps):
        org = bbox[:, :2]
        wh = bbox[:, 2:] - bbox[:, :2]
        lcl_kps = kps[:, :, :2] - np.repeat(org, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps = lcl_kps / np.repeat(wh, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps = lcl_kps.astype(np.float32)
        return lcl_kps

    @staticmethod
    def _scaling_bbox_global(bbox, frame_shape_xy):
        # xyxy -> xywh
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

        bbox = bbox.reshape(-1, 2, 2) / frame_shape_xy  # 0-1 scalling
        bbox = bbox.astype(np.float32).reshape(-1, 4)
        return bbox

    def _create_mask(self, kps):
        mask = np.where(
            kps[:, :, 2] < self._th_mask, -1e10, 0.0
        )  # -inf to nan in softmax of attention module
        seq_len, points = mask.shape
        mask = np.repeat(mask, 2, axis=1).reshape(seq_len, points, 2)
        return mask

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray, NDArray]:
        frame_nums, ids, bbox, kps = self._data[idx]

        if self._data_type == IndividualDataTypes.global_:
            # global
            glb_kps = self._scaling_keypoints_global(kps, self._frame_shape_xy)
            mask = self._create_mask(kps)
            return frame_nums, ids, glb_kps, mask
        elif self._data_type == IndividualDataTypes.global_bbox:
            # global_bbox
            glb_bbox = self._scaling_bbox_global(bbox, self._frame_shape_xy)
            return frame_nums, ids, glb_bbox, np.empty((0,))
        elif self._data_type == IndividualDataTypes.local:
            # local
            lcl_kps = self._scaling_keypoints_local(bbox, kps)
            mask = self._create_mask(kps)
            return frame_nums, ids, lcl_kps, mask
        else:
            # both
            glb_kps = self._scaling_keypoints_global(kps, self._frame_shape_xy)
            lcl_kps = self._scaling_keypoints_local(bbox, kps)
            kps = np.concatenate([glb_kps, lcl_kps], axis=1)
            mask = self._create_mask(kps)
            mask = np.repeat(mask, 2, axis=0).reshape(kps.shape)
            return frame_nums, ids, kps, mask
