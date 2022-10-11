from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class IndividualDataFormat:
    frame_num: str = "frame"
    id: str = "id"
    stage: str = "stage"
    z: str = "z"
    w_spat: str = "weights_spatial"
    w_temp: str = "weights_temporal"
    conf: str = "confidence"
    anomaly: str = "anomaly"

    @classmethod
    def get_names(cls):
        return list(cls.__dict__.keys())


@dataclass(frozen=True)
class IndividualModelTypes:
    gan: str = "gan"
    egan: str = "egan"
    autoencoder: str = "autoencoder"

    @classmethod
    def get_types(cls) -> List[str]:
        return list(cls.__dict__.keys())

    @classmethod
    def includes(cls, model_type: str):
        return model_type.casefold() in cls.get_types()
