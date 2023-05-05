import gc
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from mmpose.structures import PoseDataSample
from numpy.typing import NDArray
from tqdm import tqdm

from modules.utils.video import Capture

from ..format import Format
from .detection import Detector
from .tracking import Tracker


class PoseModel:
    def __init__(
        self,
        pose_cfg: dict,
        det_gpu: int,
        trk_gpu: int,
        with_clahe: bool = False,
    ):
        self._cfg = pose_cfg

        print("=> loading detector model")
        self._detector = Detector(self._cfg, f"cuda:{det_gpu}")
        print("=> loading tracker model")
        self._tracker = Tracker(self._cfg, f"cuda:{trk_gpu}")

        self._with_clahe = with_clahe
        if with_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __del__(self):
        del self._detector, self._tracker
        if self._with_clahe:
            del self._clahe
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _get_bboxs_from_det_results(det_results: List[PoseDataSample]):
        return np.array([result.pred_instances.bboxes[0] for result in det_results])

    @staticmethod
    def _get_kps_from_det_results(det_results: List[PoseDataSample]):
        return np.array(
            [
                np.hstack(
                    [
                        result.pred_instances.keypoints[0],
                        result.pred_instances.keypoint_scores.T,
                    ]
                )
                for result in det_results
            ]
        )

    def predict(self, cap: Capture) -> List[Dict[str, Any]]:
        print("=> running pose estimation")
        results = []
        for frame_num in tqdm(range(cap.frame_count), ncols=100):
            frame_num += 1
            frame = cap.read()[1]

            # CLAHE
            if self._with_clahe:
                cl_r = self._clahe.apply(frame[:, :, 2])
                cl_g = self._clahe.apply(frame[:, :, 1])
                cl_b = self._clahe.apply(frame[:, :, 0])
                frame = cv2.merge((cl_b, cl_g, cl_r))

            # keypoints detection
            det_results = self._detector.predict(frame)

            # extract unique result
            bboxs = self._get_bboxs_from_det_results(det_results)
            kps = self._get_kps_from_det_results(det_results)
            if len(kps) > 0:
                remain_indices = self._del_leaky(kps, self._cfg["th_delete"])
                remain_indices = self._get_unique(
                    kps,
                    remain_indices,
                    self._cfg["th_diff"],
                    self._cfg["th_count"],
                )
                bboxs = bboxs[remain_indices]
                kps = kps[remain_indices]

            # tracking
            tracks = self._tracker.update(frame, kps)

            # append result
            for t in tracks:
                # get id is closed kps
                i = np.where(np.isclose(t.pose, kps))[0]
                if len(i) == 0:
                    continue

                # select most frequent index
                # unique, freq = np.unique(i, return_counts=True)
                # i = unique[np.argmax(freq)]
                # select minimum index
                i = np.min(i)

                # create result
                result = {
                    Format.frame_num: int(frame_num),
                    Format.id: int(t.track_id),
                    Format.bbox: bboxs[i],
                    Format.keypoints: kps[i],
                }

                results.append(result)

            del det_results, kps, tracks
        gc.collect()

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
