from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from modules.individual import IndividualDataFormat
from modules.utils.keypoints import KPName
from modules.utils.video import concat_frames
from numpy.typing import NDArray

from .pose import write_frame as pose_write_frame


def write_frame_one_with_heatmap(
    frame: NDArray,
    individual_data: Dict[str, Any],
    frame_num: int,
    heatmap_length: int = 30,
) -> NDArray:
    frame = pose_write_frame(frame, [individual_data], frame_num, False)

    # weights spatial
    heatmap = _create_heatmap(
        individual_data[IndividualDataFormat.w_spat], heatmap_length
    )
    frame = concat_frames(frame, heatmap)

    # weights temporal
    heatmap = _create_heatmap(
        individual_data[IndividualDataFormat.w_temp], heatmap_length
    )
    frame = concat_frames(frame, heatmap)

    # feature
    if IndividualDataFormat.feature in individual_data:
        # G doesn't output feature, D outputs feature
        heatmap = _create_heatmap(individual_data[IndividualDataFormat.feature])
        frame = concat_frames(frame, heatmap)

    return frame


def _create_heatmap(feature: NDArray, length: int = None, **kwargs) -> NDArray:
    fig = plt.figure()
    plt.clf()
    xticklabels = [i + 1 for i in range(0, length, length // 10)]
    sns.heatmap(
        feature[:length],
        vmax=kwargs["vmax"],
        vmin=kwargs["vmin"],
        cmap="Blues",
        cbar=False,
        xticklabels=xticklabels,
        yticklabels=KPName.get_names(),
    )
    fig.canvas.draw()

    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return frame
