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
            frame_num = frame_num.cpu().numpy()[0]
            frame = frame.cpu().numpy()[0]

            # keypoints detection
            kps = self._detect_keypoints(frame)

            # tracking
            tracks = self._tracker.update(frame, kps)

            # append result
            for t in tracks:
                results.append((frame_num, t.track_id, t.pose))

            del kps, tracks

        return results

    def _detect_keypoints(self, frame):
        det_results = self._detector.predict(frame)
        kps_results = np.array([det["keypoints"] for det in det_results])
        kps_results = self._del_leaky(kps_results, self._cfg["th_delete"])
        kps_results = self._get_unique(
            kps_results, self._cfg["th_diff"], self._cfg["th_count"]
        )

        return kps_results

    @staticmethod
    def _del_leaky(kps_results: NDArray, th_delete: float):
        return kps_results[
            np.nonzero(np.mean(kps_results[:, :, 2], axis=1) > th_delete)
        ]

    @staticmethod
    def _get_unique(kps_results: NDArray, th_diff: float, th_count: int):
        unique_kps = np.empty((0, 17, 3))

        for i in range(len(kps_results)):
            found_overlap = False

            for j in range(len(unique_kps)):
                diff = np.linalg.norm(
                    kps_results[i, :, :2] - unique_kps[j, :, :2], axis=1
                )
                if len(np.where(diff < th_diff)[0]) >= th_count:
                    found_overlap = True
                    if np.mean(kps_results[i, :, 2]) > np.mean(unique_kps[j, :, 2]):
                        # select one has more confidence score
                        unique_kps[j] = kps_results[i]
                    break

            if not found_overlap:
                # if there aren't overlapped
                unique_kps = np.append(unique_kps, [kps_results[i]], axis=0)

        return unique_kps
