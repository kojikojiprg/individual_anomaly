import gc
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from torch.utils.data import Subset
from tqdm.auto import tqdm

from modules.individual import IndividualDataFormat
from modules.individual.models.datamodule import (
    AbstractIndividualDataModule,
    AbstractIndividualDataset,
)
from modules.pose import PoseDataFormat
from modules.utils.constants import Stages


class RoleEstimationDataModule(AbstractIndividualDataModule):
    def __init__(
        self,
        pose_data_dir: str,
        annotation_path: str,
        config: SimpleNamespace,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__(pose_data_dir, config, stage)

        pose_data_lst = self._load_pose_data(
            self._data_dirs, data_keys=[PoseDataFormat.bbox, PoseDataFormat.keypoints]
        )
        pose_data_lst = self._load_annotations(annotation_path, pose_data_lst)

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

    @staticmethod
    def _load_annotations(
        annotation_path: str, pose_data_lst: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        # load annotation
        annotations = np.loadtxt(annotation_path, skiprows=1, delimiter=",", dtype=int)

        for i, data in enumerate(pose_data_lst):
            video_num = i + 1

            # get annotation of video_num
            annotations_tmp = annotations[np.where(annotations[:, 0] == video_num)]
            # print(video_num)
            # print(annotations_tmp)
            for item in data:
                frame_num = item[PoseDataFormat.frame_num]
                pid = item[PoseDataFormat.id]
                if pid not in annotations_tmp[:, 1]:
                    item[IndividualDataFormat.roll_label] = 0  # no labeled
                    continue

                # print(pid)
                for an in annotations_tmp:
                    start_frame_num, end_frame_num = an[2:4]
                    if start_frame_num <= frame_num and frame_num <= end_frame_num:
                        item[IndividualDataFormat.roll_label] = an[-1]  # labeled
                    else:
                        item[IndividualDataFormat.roll_label] = 0  # no labeled
            # print(pose_data_lst[0][0])
            # raise KeyError

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


class IndividualDataset(AbstractIndividualDataset):
    def __init__(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
        stage: str,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__(pose_data, seq_len, th_split, stage, frame_shape)

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
        pre_label = pose_data[0][IndividualDataFormat.roll_label]

        seq_data: list = []
        for item in tqdm(pose_data, leave=False, ncols=100):
            # get values
            frame_num = item[PoseDataFormat.frame_num]
            pid = item[PoseDataFormat.id]
            bbox = np.array(item[PoseDataFormat.bbox])[:4]
            kps = np.array(item[PoseDataFormat.keypoints])
            label = item[IndividualDataFormat.roll_label]

            if np.any(kps[:, 2] < 0.2):
                continue

            if pid != pre_pid:
                if len(seq_data) > seq_len:
                    self._append(seq_data, seq_len)
                seq_data = []  # reset seq_data
            else:
                if (
                    1 < frame_num - pre_frame_num
                    and frame_num - pre_frame_num <= th_split
                ):
                    # fill brank with nan
                    if len(seq_data) > 0:
                        seq_data += [
                            (
                                num,
                                pid,
                                np.full((4,), np.nan),
                                np.full((17, 3), np.nan),
                                pre_label,
                            )
                            for num in range(pre_frame_num + 1, frame_num)
                        ]
                elif th_split < frame_num - pre_frame_num:
                    if len(seq_data) > seq_len:
                        self._append(seq_data, seq_len)
                    seq_data = []  # reset seq_data
                else:
                    pass

            # append keypoints to seq_data
            seq_data.append((frame_num, pid, bbox, kps, label))

            # update frame_num and id
            pre_frame_num = frame_num
            pre_pid = pid
            pre_label = label
        else:
            if len(seq_data) > seq_len:
                self._append(seq_data, seq_len)

        del seq_data
        del frame_num, pid, kps
        del pre_frame_num, pre_pid
        del pose_data
        gc.collect()

    def _append(self, seq_data, seq_len):
        # collect and kps
        all_bbox = np.array([item[2] for item in seq_data])
        all_kps = np.array([item[3] for item in seq_data])

        # interpolate bbox and keypoints
        all_bbox = self._interpolate2d(all_bbox)
        all_kps_t = all_kps.transpose(1, 0, 2)
        for i in range(len(all_kps_t)):
            all_kps_t[i] = self._interpolate2d(all_kps_t[i])
        all_kps = all_kps_t.transpose(1, 0, 2)

        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            frame_num = seq_data[i + seq_len - 1][0]
            pid = f"{seq_data[i + seq_len - 1][1]}"
            bbox = all_bbox[i : i + seq_len]
            kps = all_kps[i : i + seq_len]
            labels = seq_data[i + seq_len - 1][4]

            bbox = self._scaling_bbox(bbox)
            kps = self._scaling_keypoints(kps)

            self._data.append((frame_num, pid, bbox, kps, labels))

            if self._stage == Stages.train:
                # data augumentation by flipping horizontal
                bbox[:, :, 0] = (bbox[:, :, 0] * -1) + 1
                kps[:, :, 0] = (kps[:, :, 0] * -1) + 1
                self._data.append((frame_num, pid, bbox, kps, labels))

    @staticmethod
    def _interpolate2d(vals: NDArray):
        ret_vals = np.empty((vals.shape[0], 0))
        for i in range(vals.shape[1]):
            x = np.where(~np.isnan(vals[:, i]))[0]
            y = vals[:, i][~np.isnan(vals[:, i])]
            fitted_curve = interpolate.interp1d(x, y)
            fitted_y = fitted_curve(np.arange(len(vals)))
            fitted_y[np.isnan(vals[:, i])] += np.random.normal(
                0, 5, len(fitted_y[np.isnan(vals[:, i])])
            )

            ret_vals = np.append(ret_vals, fitted_y.reshape(-1, 1), axis=1)
        return ret_vals

    def _scaling_bbox(self, bbox):
        bbox = bbox.reshape(-1, 2, 2)
        bbox = bbox / self._frame_shape  # scalling bbox top-left
        bbox = bbox.astype(np.float32)
        return bbox

    @staticmethod
    def _scaling_keypoints(kps):
        kps = kps[:, :17]  # extract upper body
        org = np.min(kps[:, :, :2], axis=1)
        wh = np.max(kps[:, :, :2], axis=1) - org
        lcl_kps = kps[:, :, :2] - np.repeat(org, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps /= np.repeat(wh, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps = lcl_kps.astype(np.float32)
        return lcl_kps
