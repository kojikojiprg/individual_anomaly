from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class IndividualDataFormat:
    frame_num: str = "frame"
    id: str = "id"
    kps_real: str = "keypoints_real"
    kps_fake: str = "keypoints_fake"
    z: str = "z"
    attn: str = "attn_weights"
    f_real: str = "d_feature_real"
    f_fake: str = "d_feature_fake"
    loss_r: str = "loss_resitudal"
    loss_d: str = "loss_discrimination"

    @classmethod
    def get_keys(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]


@dataclass(frozen=True)
class IndividualDataTypes:
    global_: str = "global"
    local: str = "local"
    both: str = "both"
    bbox: str = "bbox"

    @classmethod
    def get_types(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]

    @classmethod
    def includes(cls, data_type: str) -> bool:
        return data_type.casefold() in cls.get_types()


@dataclass(frozen=True)
class IndividualModelTypes:
    egan: str = "egan"
    ganomaly: str = "ganomaly"

    @classmethod
    def get_types(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]

    @classmethod
    def includes(cls, model_type: str) -> bool:
        return model_type.casefold() in cls.get_types()
