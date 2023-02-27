import yaml

from .data_handler import PoseDataHandler
from .format import Format as PoseDataFormat
from .model import PoseModel


class PoseEstimation:
    def __init__(
        self,
        cfg_path: str,
        det_gpu: int,
        trk_gpu: int,
    ):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self._model = PoseModel(cfg, det_gpu, trk_gpu)

    def __del__(self):
        del self._model

    def inference(self, video_path: str, data_dir: str = None):
        cap = PoseDataHandler.create_video_capture(video_path)

        results = self._model.predict(cap)
        if data_dir is not None:
            print(f"=> saving pose estimation results to {data_dir}")
            PoseDataHandler.save(data_dir, results)
            PoseDataHandler.save_frame_shape(data_dir, cap.size)

        return results
