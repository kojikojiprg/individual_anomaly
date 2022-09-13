from logging import Logger
from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from .detection import Detector
from .tracking import Tracker


class KPModel:
    def __init__(self, pose_cfg: dict, device: str, logger: Logger):
        self._logger = logger

        self._cfg = pose_cfg

        self._logger.info("=> loading detector model")
        self._detector = Detector(self._cfg, device)
        self._logger.info("=> loading tracker model")
        self._tracker = Tracker(self._cfg, device)

    def __del__(self):
        torch.cuda.empty_cache()
        del self._detector, self._tracker, self._logger

    def predict(
        self, data_loader: torch.utils.data.DataLoader
    ) -> List[Tuple[int, int, NDArray]]:
        self._logger.info("=> estimating keypoints")
        results = []
        for frame_num, frame in tqdm(data_loader, ncols=100):
            kps = self._detect(frame)

            # tracking
            tracks = self._tracker.update(frame, kps)

            # append result
            for t in tracks:
                results.append((frame_num, t.track_id, np.array(t.pose)))

            del kps, tracks

        return results

    def _detect(self, frame):
        # keypoints detection
        kps_all = self._detector.predict(frame.cpu().numpy())
        for kps in kps_all:
            kps = self._del_leaky(kps, self._cfg["th_delete"])
            kps = self._get_unique(kps, self._cfg["th_diff"], self._cfg["th_count"])
            kps_all.append(kps)

        return kps_all

    @staticmethod
    def _del_leaky(kps: NDArray, th_delete: float):
        return kps[np.nonzero(np.mean(kps[:, :, 2], axis=1) > th_delete)]

    @staticmethod
    def _get_unique(kps: NDArray, th_diff: float, th_count: int):
        unique_kps = np.empty((0, 17, 3))

        for i in range(len(kps)):
            found_overlap = False

            for j in range(len(unique_kps)):
                diff = np.linalg.norm(kps[i, :, :2] - unique_kps[j, :, :2], axis=1)
                if len(np.where(diff < th_diff)[0]) >= th_count:
                    found_overlap = True
                    if np.mean(kps[i, :, 2]) > np.mean(unique_kps[j, :, 2]):
                        # select one has more confidence score
                        unique_kps[j] = kps[i]
                    break

            if not found_overlap:
                # if there aren't overlapped
                unique_kps = np.append(unique_kps, [kps[i]], axis=0)

        return unique_kps
