from dataclasses import dataclass
from typing import List, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KPName:
    Nose: int = 0
    LEye: int = 1
    REye: int = 2
    LEar: int = 3
    REar: int = 4
    LShoulder: int = 5
    RShoulder: int = 6
    LElbow: int = 7
    RElbow: int = 8
    LWrist: int = 9
    RWrist: int = 10
    LHip: int = 11
    RHip: int = 12
    LKnee: int = 13
    RKnee: int = 14
    LAnkle: int = 15
    RAnkle: int = 16


class KPClass:
    def __init__(self, keypoints: Union[list, NDArray]):
        keypoints = np.array(keypoints)
        if keypoints.shape != (17, 3):
            keypoints = keypoints.reshape(17, 3)

        self._keypoints = keypoints

    def get(self, name: int, ignore_confidence: bool = False) -> NDArray:
        if ignore_confidence:
            return self._keypoints[name][:2].copy()
        else:
            return self._keypoints[name].copy()

    def get_middle(self, name, th_conf=None) -> Union[NDArray, None]:
        if th_conf is None:
            th_conf = 0.0

        R = self.get(getattr(KPName, "R" + name))
        L = self.get(getattr(KPName, "L" + name))
        if R[2] < th_conf and L[2] < th_conf:
            return None
        elif R[2] < th_conf:
            point = L
        elif L[2] < th_conf:
            point = R
        else:
            point = (R + L) / 2
        return point[:2].astype(np.int32)

    def ravel(self) -> NDArray:
        return self._keypoints.ravel()


class KPListClass(list):
    def __init__(self):
        super().__init__([])

    def get_middle_points(self, name: str) -> List[NDArray]:
        points = []
        for keypoints in self:
            if keypoints is not None:
                points.append(keypoints.get_middle(name))
            else:
                points.append(None)

        return points

    def append(self, keypoints: Union[KPClass, NDArray]):
        if type(keypoints) == KPClass:
            super().append(keypoints)
        elif type(keypoints) == NDArray:
            if keypoints.shape == (17, 3) or keypoints.shape == (51,):
                super().append(KPClass(keypoints))
            else:
                raise TypeError
        else:
            raise TypeError
