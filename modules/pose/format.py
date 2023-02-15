from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    frame_num: str = "frame"
    id: str = "id"
    bbox: str = "bbox"
    keypoints: str = "keypoints"
    heatmap: str = "heatmap"
