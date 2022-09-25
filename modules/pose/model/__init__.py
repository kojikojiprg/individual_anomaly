from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch
from modules.utils.video import Capture
from numpy.typing import NDArray
from tqdm import tqdm

from ..format import Format
from .detection import Detector
from .tracking import Tracker


class PoseModel:
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
        self, cap: Capture, return_heatmap: bool = False
    ) -> List[Dict[str, Any]]:
        self._logger.info("=> running pose estimation")
        results = []
        for frame_num in tqdm(range(cap.frame_count), ncols=100):
            frame_num += 1
            frame = cap.read()[1]

            # keypoints detection
            det_results, heatmaps = self._detector.predict(frame, return_heatmap)

            # extract unique result
            kps = np.array([det["keypoints"] for det in det_results])
            if len(kps) > 0:
                remain_indices = self._del_leaky(kps, self._cfg["th_delete"])
                remain_indices = self._get_unique(
                    kps,
                    remain_indices,
                    self._cfg["th_diff"],
                    self._cfg["th_count"],
                )

            # tracking
            kps = kps[remain_indices]
            tracks = self._tracker.update(frame, kps)

            # append result
            for t in tracks:
                # select index
                i = np.where(np.isclose(t.pose, kps))[0]
                if len(i) == 0:
                    continue
                unique, freq = np.unique(i, return_counts=True)
                i = unique[np.argmax(freq)]
                i = remain_indices[i]

                # create result
                result = {
                    Format.frame_num: frame_num,
                    Format.id: t.track_id,
                    Format.bbox: det_results[i][Format.bbox],
                    Format.keypoints: det_results[i][Format.keypoints],
                }
                if return_heatmap:
                    result[Format.heatmap] = heatmaps[i]

                results.append(result)

            del det_results, kps, tracks, heatmaps

        return results

    @staticmethod
    def _del_leaky(kps: NDArray, th_delete: float):
        return np.where(np.mean(kps[:, :, 2], axis=1) >= th_delete)[0]

    @staticmethod
    def _get_unique(kps: NDArray, indices: NDArray, th_diff: float, th_count: int):
        remain_indices = np.empty((0,), dtype=np.int32)

        for idx in indices:
            for ridx in remain_indices:
                # calc diff of all points
                diff = np.linalg.norm(kps[idx, :, :2] - kps[ridx, :, :2], axis=1)

                if len(np.where(diff < th_diff)[0]) >= th_count:
                    # found overlap
                    if np.mean(kps[idx, :, 2]) > np.mean(kps[ridx, :, 2]):
                        # select one which is more confidence
                        remain_indices[remain_indices == ridx] = idx

                    break
            else:
                # if there aren't overlapped
                remain_indices = np.append(remain_indices, idx)

        # return unique_kps
        return remain_indices
