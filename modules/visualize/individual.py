import gc
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from modules import DataFormat
from modules.utils.video import Capture, Writer

from .pose import draw_bbox, put_frame_num

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


class IndividualVisualizer:
    def __init__(self, args=None):
        pass

    def visualise(self, video_path: str, data_dir: str, ind_data_lst: str):
        # create video capture
        print(f"=> loading video from {video_path}.")
        video_capture = Capture(video_path)
        assert (
            video_capture.is_opened
        ), f"{video_path} does not exist or is wrong file type."

        tmp_frame = video_capture.read()[1]
        video_capture.set_pos_frame_count(0)

        video_num = os.path.basename(video_path).split(".")[0]

        # create video writer
        out_path = os.path.join(data_dir, f"{video_num}_pose.mp4")
        video_writer = Writer(out_path, video_capture.fps, tmp_frame.shape[1::-1])

        # create video writer for individual results
        # if self._do_individual:
        #     out_path = os.path.join(data_dir, f"individual_{}.mp4")

        #     pose_video_writer = Writer(
        #         out_path, video_capture.fps, tmp_frame.shape[1::-1]
        #     )
        #     out_paths.append(out_path)

        print(f"=> writing video into {out_path}.")
        for frame_num in tqdm(range(video_capture.frame_count), ncols=100):
            frame_num += 1  # frame_num = (1, ...)
            ret, frame = video_capture.read()

            # write pose estimation video
            frame = write_frame(frame, ind_data_lst, frame_num)
            video_writer.write(frame)

        # release memory
        del video_capture
        del video_writer
        gc.collect()


def write_frame(frame, ind_data_lst, frame_num):
    # add keypoints to image
    frame = put_frame_num(frame, frame_num)
    for data in ind_data_lst:
        if data[DataFormat.frame_num] == frame_num:
            if data[DataFormat.role_aux] == 1:
                frame = draw_bbox(frame, np.array(data[DataFormat.bbox_real]))

    return frame


def _plot_val_kps(img, kps, color):
    for i in range(len(kps) - 1):
        for j in range(i + 1, len(kps)):
            if graph[i][j] == 1:
                p1 = tuple(kps[i].astype(int))
                p2 = tuple(kps[j].astype(int))
                img = cv2.line(img, p1, p2, color, 3)
    return img


def plot_val_kps(
    kps_real,
    kps_fake,
    pid,
    epoch,
    data_type,
    seq_len=10,
    plot_size=(300, 400),
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


def plot_val_bbox(
    bbox_real, bbox_fake, pid, epoch, data_type, seq_len=10, plot_size=(300, 400)
):
    fig = plt.figure(figsize=(20, 3))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)

    step = len(bbox_real) // seq_len
    bbox_real = bbox_real[step - 1 :: step]
    bbox_fake = bbox_fake[step - 1 :: step]

    mins = np.min(
        np.append(np.min(bbox_real, axis=1), np.min(bbox_fake, axis=1), axis=0), axis=0
    )
    maxs = np.max(
        np.append(np.max(bbox_real, axis=1), np.max(bbox_fake, axis=1), axis=0), axis=0
    )
    size = maxs - mins
    ratio = np.array(plot_size) / size
    bbox_real = (bbox_real - mins) * ratio
    bbox_fake = (bbox_fake - mins) * ratio

    for j in range(seq_len):
        img = np.full((plot_size[1], plot_size[0], 3), 255, np.uint8)
        img = _plot_val_kps(img, bbox_real[j], (0, 255, 0))  # real: green
        img = _plot_val_kps(img, bbox_fake[j], (255, 0, 0))  # fake: red
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
