import gc
from typing import List

import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine import init_default_scope
from mmengine.config import Config
from mmpose.apis import inference_bottomup, inference_topdown, init_model
from mmpose.evaluation.functional import nms
from mmpose.structures import PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline
from numpy.typing import NDArray


class Detector:
    def __init__(self, cfg: dict, device: str):
        self._cfg = cfg
        self._device = device

        # build the pose model from a config file and a checkpoint file
        init_default_scope("mmpose")
        mmpose_cfg = Config.fromfile(cfg["configs"]["mmpose"])
        self._pose_model = init_model(
            mmpose_cfg.config, mmpose_cfg.weights, device=device
        )
        self._model_type = mmpose_cfg.type

        self._det_model = None
        if self._model_type == "top-down":
            # build the detection model from a config file and a checkpoint file
            init_default_scope("mmdet")
            mmdet_cfg = Config.fromfile(cfg["configs"]["mmdet"])
            self._det_model = init_detector(
                mmdet_cfg.config, mmdet_cfg.weights, device=device
            )
            self._det_model.cfg = adapt_mmdet_pipeline(self._det_model.cfg)
        init_default_scope("mmpose")

    def __del__(self):
        del self._det_model, self._pose_model
        gc.collect()

    def predict(self, img: NDArray):
        if self._model_type == "top-down":
            return self.predict_top_down(img)
        else:
            return self.predict_bottom_up(img)

    def _process_mmdet_results(self, det_result, cat_id, th_bbox, th_iou):
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            np.logical_and(
                pred_instance.labels == cat_id,
                pred_instance.scores > th_bbox,
            )
        ]
        bboxes = bboxes[nms(bboxes, th_iou), :4]
        return bboxes

    def predict_top_down(self, img: NDArray) -> List[PoseDataSample]:
        det_results = inference_detector(self._det_model, img)
        bboxes = self._process_mmdet_results(
            det_results,
            cat_id=0,
            th_bbox=self._cfg["th_bbox"],
            th_iou=self._cfg["th_iou"],
        )

        pose_results = inference_topdown(
            self._pose_model,
            img,
            bboxes,
            bbox_format="xyxy",
        )

        return pose_results

    def predict_bottom_up(self, img: NDArray):
        # TODO: modify for updating
        pose_results, heatmaps = inference_bottomup(
            self._pose_model,
            img,
            dataset=self._pose_model.cfg.data.test.type,
        )

        return pose_results, heatmaps
