from logging import Logger
from typing import Any, Dict, List, Tuple

import numpy as np
from modules.pose import PoseDataFormat
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm import tqdm


class IndividualDataset(Dataset):
    _dmy_kps = np.full((17, 3), np.nan, dtype=np.float32)

    def __init__(
        self,
        pose_data_lst: List[List[Dict[str, Any]]],
        seq_len: int,
        th_split: int,
        logger: Logger,
    ):
        super().__init__()

        self._data: List[Tuple[int, NDArray]] = []
        self._logger = logger

        self.create_dataset(pose_data_lst, seq_len, th_split)

    def create_dataset(
        self,
        pose_data_lst: List[List[Dict[str, Any]]],
        seq_len: int,
        th_split: int,
    ):
        self._logger.info("=> creating dataset")
        for pose_data in tqdm(pose_data_lst, ncols=100):
            # sort data by id
            pose_data = sorted(pose_data, key=lambda x: x[PoseDataFormat.id])

            # get frame_num and id of first data
            pre_frame_num = pose_data[0][PoseDataFormat.frame_num]
            pre_id = pose_data[0][PoseDataFormat.id]

            seq_data: list = []
            for item in pose_data:
                # get values
                frame_num = item[PoseDataFormat.frame_num]
                id = item[PoseDataFormat.id]
                keypoints = item[PoseDataFormat.keypoints]

                if id != pre_id:
                    if len(seq_data) > seq_len:
                        # append data with creating sequential data
                        for i in range(0, len(seq_data) - seq_len + 1):
                            self._data.append((id, np.array(seq_data[i : i + seq_len])))
                    # reset seq_data
                    seq_data = []
                else:
                    if (
                        1 < frame_num - pre_frame_num
                        and frame_num - pre_frame_num <= th_split
                    ):
                        # fill brank with nan
                        seq_data += [
                            self._dmy_kps for _ in range(frame_num - pre_frame_num)
                        ]
                    elif th_split < frame_num - pre_frame_num:
                        if len(seq_data) > seq_len:
                            # append data with creating sequential data
                            for i in range(0, len(seq_data) - seq_len + 1):
                                self._data.append(
                                    (id, np.array(seq_data[i : i + seq_len]))
                                )
                        # reset seq_data
                        seq_data = []
                    else:
                        pass

                # append keypoints to seq_data
                seq_data.append(keypoints)

                # update frame_num and id
                pre_frame_num = frame_num
                pre_id = id

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[int, NDArray]:
        id, keypoints = self._data[idx]
        return id, keypoints[:, :, :2]
