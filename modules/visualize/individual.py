import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from modules.individual.constants import IndividualDataTypes

graph = [
    # ========== 4 ============ 9 =========== 14 =====
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nose
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LEye
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # REye
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LEar
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # REar
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # LShoulder
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # RShoulder
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # LElbow
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # RElbow
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LWrist
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RWrist
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # LHip
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # RHip
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # LKnee
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # RKnee
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LAnkle
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RAnkle
]


def _plot_val_kps(img, kps, color):
    for i in range(len(kps) - 1):
        for j in range(i + 1, len(kps)):
            if graph[i][j] == 1:
                p1 = tuple(kps[i].astype(int))
                p2 = tuple(kps[j].astype(int))
                img = cv2.line(img, p1, p2, color, 3)
    return img


def plot_val_kps(
    kps_real, kps_fake, pid, epoch, data_type, seq_len=10, plot_size=(300, 400)
):
    fig = plt.figure(figsize=(20, 3))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)

    step = len(kps_real) // seq_len
    kps_real = kps_real[step - 1 :: step]
    kps_fake = kps_fake[step - 1 :: step]

    mins = np.min(
        np.append(np.min(kps_real, axis=1), np.min(kps_fake, axis=1), axis=0), axis=0
    )
    maxs = np.max(
        np.append(np.max(kps_real, axis=1), np.max(kps_fake, axis=1), axis=0), axis=0
    )
    size = maxs - mins
    ratio = np.array(plot_size) / size
    kps_real = (kps_real - mins) * ratio
    kps_fake = (kps_fake - mins) * ratio

    if data_type == IndividualDataTypes.local_bbox:
        kps_real = kps_real[:, 1:]  # idx0 = bbox top-left
        kps_fake = kps_fake[:, 1:]

    for j in range(seq_len):
        img = np.full((plot_size[1], plot_size[0], 3), 255, np.uint8)
        img = _plot_val_kps(img, kps_real[j], (0, 255, 0))  # real: green
        img = _plot_val_kps(img, kps_fake[j], (255, 0, 0))  # fake: red
        ax = fig.add_subplot(1, seq_len, j + 1)
        ax.imshow(img)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # save fig
    path = os.path.join(
        "data",
        "images",
        "individual",
        "ganomaly",
        "generator",
        data_type,
        f"pid{pid}_epoch{epoch}.jpg",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")


def plot_heatmap(data, pid, epoch, name, vmin=None, vmax=None):
    plt.figure()
    sns.heatmap(data, vmin=vmin, vmax=vmax)
    path = os.path.join(
        "data",
        "images",
        "individual",
        "ganomaly",
        "generator",
        f"pid{pid}_epoch{epoch}_{name}.jpg",
    )
    plt.savefig(path, bbox_inches="tight")
