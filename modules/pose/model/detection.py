from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    inference_bottom_up_pose_model,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)
from numpy.typing import NDArray


class Detector:
    def __init__(self, cfg: dict, device: str):
        self._cfg = cfg
        self._device = device

        # build the pose model from a config file and a checkpoint file
        mmpose_cfg = Config.fromfile(cfg["configs"]["mmpose"])
        self._pose_model = init_pose_model(
            mmpose_cfg.config, mmpose_cfg.weights, device=device
        )
        self._model_type = mmpose_cfg.type

        self._det_model = None
        if self._model_type == "top-down":
            # build the detection model from a config file and a checkpoint file
            mmdet_cfg = Config.fromfile(cfg["configs"]["mmdet"])
            self._det_model = init_detector(
                mmdet_cfg.config, mmdet_cfg.weights, device=device
            )

    def __del__(self):
        del self._det_model, self._pose_model

    def predict(self, img: NDArray, return_heatmap: bool = False):
        if self._model_type == "top-down":
            return self.predict_top_down(img, return_heatmap)
        else:
            return self.predict_bottom_up(img, return_heatmap)

    def predict_top_down(self, img: NDArray, return_heatmap: bool = False):
        mmdet_results = inference_detector(self._det_model, img)

        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, heatmaps = inference_top_down_pose_model(
            self._pose_model,
            img,
            person_results,
            bbox_thr=self._cfg["th_bbox"],
            format="xyxy",
            dataset=self._pose_model.cfg.data.test.type,
            return_heatmap=return_heatmap,
        )

        if return_heatmap:
            heatmaps = heatmaps[0]["heatmap"]

        return pose_results, heatmaps

    def predict_bottom_up(self, img: NDArray, return_heatmap: bool = False):
        pose_results, heatmaps = inference_bottom_up_pose_model(
            self._pose_model,
            img,
            dataset=self._pose_model.cfg.data.test.type,
            return_heatmap=return_heatmap,
        )

        if return_heatmap:
            heatmaps = heatmaps[0]["heatmap"]

        return pose_results, heatmaps
