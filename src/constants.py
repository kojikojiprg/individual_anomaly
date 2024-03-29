from dataclasses import dataclass
from typing import List


class _AbstractDataClass:
    @classmethod
    def get_attributes(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]

    @classmethod
    def includes(cls, attribute: str) -> bool:
        return attribute.casefold() in cls.get_attributes()


@dataclass(frozen=True)
class DataFormat(_AbstractDataClass):
    frame_num: str = "frame"
    id: str = "id"

    bbox_real: str = "bbox_real"
    bbox_fake: str = "bbox_fake"
    kps_real: str = "keypoints_real"
    kps_fake: str = "keypoints_fake"

    z: str = "z"
    attn: str = "attn_weights"

    f_real: str = "d_feature_real"
    f_fake: str = "d_feature_fake"

    loss_r: str = "loss_resitudal"
    loss_d: str = "loss_discrimination"

    role_label: str = "role_label"
    role_adv: str = "role_adv"
    role_aux: str = "role_aux"


@dataclass(frozen=True)
class DataTypes(_AbstractDataClass):
    global_: str = "global"
    local: str = "local"
    bbox: str = "bbox"
    bbox_local: str = "bbox+local"
