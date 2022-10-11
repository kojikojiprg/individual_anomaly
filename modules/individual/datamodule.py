import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from modules.pose import PoseDataFormat, PoseDataHandler
from modules.utils import video
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class IndividualDataModule(LightningDataModule):
    def __init__(self, data_dir: str, config: SimpleNamespace, stage: str = None):
        super().__init__()
        self._config = config
        self._stage = stage

        if stage is None or stage == "train":
            train_data_dirs = glob(os.path.join(data_dir, "train", "*"))
            self._train_pose_data_lst = self._load_pose_data(train_data_dirs)
            frame_shape = self._get_frame_shape(train_data_dirs)
            self._train_dataset = self._create_dataset(
                self._train_pose_data_lst, frame_shape, "train"
            )
        if stage is None or stage == "test":
            test_data_dirs = glob(os.path.join(data_dir, "test", "*"))
            self._test_pose_data_lst = self._load_pose_data(test_data_dirs)
            frame_shape = self._get_frame_shape(test_data_dirs)
            self._test_dataset = self._create_dataset(
                self._test_pose_data_lst, frame_shape, "test"
            )

    def train_dataloader(self, batch_size: int = None, shuffle: bool = True):
        assert self._stage is None or self._stage == "train"
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(
            self._train_dataset, batch_size, shuffle=shuffle, num_workers=8
        )

    def test_dataloader(self, batch_size: int = None):
        assert self._stage is None or self._stage == "test"
        if batch_size is None:
            batch_size = self._config.batch_size
        return DataLoader(self._test_dataset, batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self, batch_size: int = None):
        assert self._stage is None
        return [
            self.train_dataloader(batch_size, shuffle=False),
            self.test_dataloader(batch_size),
        ]

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
        video_path = data_dirs[0].replace("data/", "/raid6/surgery-video/") + ".mp4"
        cap = video.Capture(video_path)
        frame_shape = cap.size
        del cap
        return frame_shape

    def _create_dataset(
        self,
        pose_data_lst: List[List[Dict[str, Any]]],
        frame_shape: Tuple[int, int],
        stage: str,
    ):
        return IndividualDataset(
            pose_data_lst,
            self._config.seq_len,
            self._config.th_split,
            frame_shape,
            stage,
        )


class IndividualDataset(Dataset):
    # _dmy_kps = np.full((17, 3), np.nan, dtype=np.float32)

    def __init__(
        self,
        pose_data_lst: List[List[Dict[str, Any]]],
        seq_len: int,
        th_split: int,
        frame_shape_xy: Tuple[int, int],
        stage: str,
    ):
        super().__init__()

        self._stage = stage
        self._data: List[Tuple[int, int, NDArray]] = []

        for video_num, pose_data in enumerate(tqdm(pose_data_lst, desc=self._stage)):
            video_num += 1
            self._create_dataset(
                video_num, pose_data, seq_len, th_split, frame_shape_xy
            )

    def _create_dataset(
        self,
        video_num,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
        frame_shape_xy: Tuple[int, int],
    ):
        # sort data by frame_num
        pose_data = sorted(pose_data, key=lambda x: x[PoseDataFormat.frame_num])
        # sort data by id
        pose_data = sorted(pose_data, key=lambda x: x[PoseDataFormat.id])

        # get frame_num and id of first data
        pre_frame_num = pose_data[0][PoseDataFormat.frame_num]
        pre_pid = pose_data[0][PoseDataFormat.id]
        pre_kps = pose_data[0][PoseDataFormat.keypoints]

        seq_data: list = []
        for item in pose_data:
            # get values
            frame_num = item[PoseDataFormat.frame_num]
            pid = item[PoseDataFormat.id]
            keypoints = item[PoseDataFormat.keypoints]

            if pid != pre_pid:
                if len(seq_data) > seq_len:
                    self._append(video_num, seq_data, seq_len)
                # reset seq_data
                seq_data = []
            else:
                if (
                    1 < frame_num - pre_frame_num
                    and frame_num - pre_frame_num <= th_split
                ):
                    # fill brank with nan
                    seq_data += [
                        (num, pid, pre_kps)
                        for num in range(pre_frame_num + 1, frame_num)
                    ]
                elif th_split < frame_num - pre_frame_num:
                    if len(seq_data) > seq_len:
                        self._append(video_num, seq_data, seq_len)
                    # reset seq_data
                    seq_data = []
                else:
                    pass

            # append keypoints to seq_data
            keypoints = keypoints[:, :2] / frame_shape_xy  # 0-1 scalling
            keypoints = keypoints.astype(np.float32)
            seq_data.append((frame_num, pid, keypoints))

            # update frame_num and id
            pre_frame_num = frame_num
            pre_pid = pid
            pre_kps = keypoints
        else:
            self._append(video_num, seq_data, seq_len)

    def _append(self, video_num, seq_data, seq_len):
        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            self._data.append(
                (
                    seq_data[i + seq_len - 1][0],
                    f"{self._stage}_{video_num:02d}_{seq_data[i + seq_len - 1][1]}",
                    np.array([item[2] for item in seq_data[i : i + seq_len]])[:, :, :2],
                )
            )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, int, NDArray]:
        frame_nums, ids, keypoints = self._data[idx]
        return frame_nums, ids, keypoints
