from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)
from numpy.typing import NDArray


class Detector:
    def __init__(self, cfg: dict, device: str):
        self._cfg = cfg
        self._device = device

        # build the detection model from a config file and a checkpoint file
        mmdet_cfg = Config.fromfile(cfg["configs"]["mmdet"])
        self._det_model = init_detector(
            mmdet_cfg.config, mmdet_cfg.weights, device=device
        )
        # build the pose model from a config file and a checkpoint file
        mmpose_cfg = Config.fromfile(cfg["configs"]["mmpose"])
        self._pose_model = init_pose_model(
            mmpose_cfg.config, mmpose_cfg.weights, device=device
        )

    def __del__(self):
        del self._det_model, self._pose_model

    def predict(self, img: NDArray):
        mmdet_results = inference_detector(self._det_model, img)

        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, _ = inference_top_down_pose_model(
            self._pose_model,
            img,
            person_results,
            bbox_thr=self._cfg["th_bbox"],
            format="xyxy",
            dataset=self._pose_model.cfg.data.test.type,
        )

        return pose_results
