import sys
from types import SimpleNamespace
from typing import Dict, List

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from torchvision.transforms import transforms as T

sys.path.append("submodules")
from unitrack.tracker.mot.pose import PoseAssociationTracker


class Tracker:
    def __init__(self, cfg: dict, device: str):
        # set config
        opts = SimpleNamespace(**{})
        with open(cfg["configs"]["unitrack"]) as f:
            common_args = yaml.safe_load(f)
        for k, v in common_args["common"].items():
            setattr(opts, k, v)
        for k, v in common_args["posetrack"].items():
            setattr(opts, k, v)
        opts.return_stage = 2
        opts.device = device  # assign device
        self.use_lab = getattr(opts, "use_lab", False)

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(opts.im_mean, opts.im_std)]
        )

        self.tracker = PoseAssociationTracker(opts)

    def __del__(self):
        del self.tracker, self.transforms

    def update(self, img: NDArray, kps: NDArray):
        process_img = img.copy()
        if self.use_lab:
            process_img = cv2.cvtColor(process_img, cv2.COLOR_BGR2LAB)
            process_img = np.array([process_img[:, :, 0]] * 3)
            process_img = process_img.transpose(1, 2, 0)

        # Normalize RGB
        process_img = process_img / 255.0
        process_img = process_img[:, :, ::-1]
        process_img = np.ascontiguousarray(process_img)
        process_img = self.transforms(process_img)

        obs = [self._cvt_kp2ob(kp) for kp in kps]

        tracks = self.tracker.update(process_img, img, obs)
        for t in tracks:
            if not isinstance(t.pose[0], list):
                t.pose = self._cvt_ob2kp(t.pose)
            else:
                tracks.remove(t)  # remove not updated track

        return tracks

    @staticmethod
    def _cvt_kp2ob(kp: NDArray):
        # https://github.com/leonid-pishchulin/poseval
        return [
            {"id": [i], "x": [pt[0]], "y": [pt[1]], "score": [pt[2]]}
            for i, pt in enumerate(kp)
        ]

    @staticmethod
    def _cvt_ob2kp(ob: List[Dict[str, list]]):
        return [[pt["x"][0], pt["y"][0], pt["score"][0]] for pt in ob]
