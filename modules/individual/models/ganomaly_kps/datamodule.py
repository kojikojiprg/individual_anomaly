import gc
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from torch.utils.data import Subset
from tqdm.auto import tqdm

from modules.individual.constants import IndividualDataTypes
from individual.models.datamodule import (
    AbstractIndividualDataModule,
    AbstractIndividualDataset,
)
from modules.pose import PoseDataFormat
from modules.utils.constants import Stages


class IndividualDataModuleKps(AbstractIndividualDataModule):
    def __init__(
        self,
        data_dir: str,
        config: SimpleNamespace,
        data_type: str = IndividualDataTypes.local,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
    ):
        super().__init__(data_dir, config, stage)

        pose_data_lst = self._load_pose_data(
            self._data_dirs, data_keys=[PoseDataFormat.keypoints]
        )

        assert data_type in [IndividualDataTypes.local, IndividualDataTypes.global_]
        if data_type == IndividualDataTypes.global_:
            assert frame_shape is not None

        if stage == Stages.train:
            pose_data = []
            for data in pose_data_lst:
                pose_data += data
            self._train_dataset = self._create_dataset(
                pose_data, data_type, frame_shape
            )
            val_ids = np.random.randint(
                len(self._train_dataset), size=self._config.n_val
            )
            self._val_dataset = Subset(self._train_dataset, val_ids)
        elif stage == Stages.test or stage == Stages.inference:
            self._test_datasets = []
            for pose_data in tqdm(pose_data_lst):
                self._test_datasets.append(
                    self._create_dataset(pose_data, data_type, frame_shape)
                )
        else:
            raise NameError

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
            self._stage,
            frame_shape,
        )


class IndividualDataset(AbstractIndividualDataset):
    def __init__(
        self,
        pose_data: List[Dict[str, Any]],
        seq_len: int,
        th_split: int,
        th_mask: float,
        data_type: str,
        stage: str,
        frame_shape: Tuple[int, int] = None,
    ):
        self._th_mask = th_mask
        self._data_type = data_type
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

        seq_data: list = []
        for item in tqdm(pose_data, leave=False, ncols=100):
            # get values
            frame_num = item[PoseDataFormat.frame_num]
            pid = item[PoseDataFormat.id]
            kps = np.array(item[PoseDataFormat.keypoints])

            # if self._stage == Stages.train and np.any(kps[:, 2] < 0.2):
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
                            (num, pid, np.full((17, 3), np.nan))
                            for num in range(pre_frame_num + 1, frame_num)
                        ]
                elif th_split < frame_num - pre_frame_num:
                    if len(seq_data) > seq_len:
                        self._append(seq_data, seq_len)
                    seq_data = []  # reset seq_data
                else:
                    pass

            # replace nan into low confidence points
            # kps[np.where(kps[:, 2] < 0.2)[0]] = np.nan

            # append keypoints to seq_data
            seq_data.append((frame_num, pid, kps))

            # update frame_num and id
            pre_frame_num = frame_num
            pre_pid = pid
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
        all_kps = np.array([item[2] for item in seq_data])

        # delete last nan
        # print(len(all_kps), all_kps[-1, -1])
        # print(np.where(np.all(np.isnan(all_kps), axis=(1, 2))))
        # all_kps = all_kps[:max(np.where(np.all(~np.isnan(all_kps), axis=(1, 2)))[0]) + 1]
        # print(len(all_kps), all_kps[-1, -1])

        # interpolate and keypoints
        all_kps_t = all_kps.transpose(1, 0, 2)
        for i in range(len(all_kps_t)):
            all_kps_t[i] = self._interpolate2d(all_kps_t[i])
        all_kps = all_kps_t.transpose(1, 0, 2)

        # append data with creating sequential data
        for i in range(0, len(seq_data) - seq_len + 1):
            frame_num = seq_data[i + seq_len - 1][0]
            pid = f"{seq_data[i + seq_len - 1][1]}"
            kps = all_kps[i : i + seq_len]

            mask = self._create_mask(kps)
            if self._data_type == IndividualDataTypes.global_:
                # global
                kps = self._scaling_keypoints_global(kps, self._frame_shape)
            elif self._data_type == IndividualDataTypes.local:
                # local
                kps = self._scaling_keypoints_local(kps)
            else:
                raise KeyError

            self._data.append((frame_num, pid, kps, mask))

            if self._stage == Stages.train:
                # data augumentation by flipping horizontal
                kps[:, :, 0] = (kps[:, :, 0] * -1) + 1
                self._data.append((frame_num, pid, kps, mask))

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

    @staticmethod
    def _scaling_keypoints_global(kps, frame_shape):
        glb_kps = kps[:, :, :2] / frame_shape  # 0-1 scalling
        glb_kps = glb_kps.astype(np.float32)
        return glb_kps

    @staticmethod
    def _scaling_keypoints_local(kps):
        kps = kps[:, :17]  # extract upper body
        org = np.min(kps[:, :, :2], axis=1)
        wh = np.max(kps[:, :, :2], axis=1) - org
        lcl_kps = kps[:, :, :2] - np.repeat(org, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps /= np.repeat(wh, 17, axis=0).reshape(-1, 17, 2)
        lcl_kps = lcl_kps.astype(np.float32)
        return lcl_kps

    def _create_mask(self, kps):
        mask = np.where(np.mean(kps[:, :, 2], axis=1) < self._th_mask, -1e10, 0.0)
        # mask = np.repeat(mask, 2, axis=1).reshape(mask.shape[0], mask.shape[1], 2)
        return mask
