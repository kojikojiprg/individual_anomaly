from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    frame_num: str = "frame"
    id: str = "id"
    keypoints: str = "keypoints"
    feature: str = "features"
    w_spat: str = "weights_spatial"
    w_temp: str = "weights_temporal"
