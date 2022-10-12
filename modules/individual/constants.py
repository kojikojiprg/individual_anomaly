from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class IndividualDataFormat:
    frame_num: str = "frame"
    id: str = "id"
    kps_real: str = "keypoints_real"
    kps_fake: str = "keypoints_fake"
    z: str = "z"
    w_spat: str = "weights_spatial"
    w_temp: str = "weights_temporal"
    f_real: str = "d_feature_real"
    f_fake: str = "d_feature_fake"
    anomaly: str = "anomaly"

    @classmethod
    def get_keys(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]


@dataclass(frozen=True)
class IndividualModelTypes:
    gan: str = "gan"
    egan: str = "egan"
    autoencoder: str = "autoencoder"

    @classmethod
    def get_types(cls) -> List[str]:
        return [cls.__dict__[key] for key in cls.__annotations__.keys()]

    @classmethod
    def includes(cls, model_type: str) -> bool:
        return model_type.casefold() in cls.get_types()
