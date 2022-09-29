from dataclasses import dataclass


@dataclass(frozen=True)
class DataFormat:
    frame: str = "frame"
    id: str = "id"
    keypoints: str = "keypoints"
