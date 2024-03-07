#!/usr/bin/env python
import argparse
import gc
import os
import sys
import warnings
from glob import glob
from typing import Any, Dict, List

import numpy as np
from scipy import interpolate
from tqdm import tqdm

sys.path.append(".")
from modules.utils import json_handler
from modules.utils.video import Capture, Writer
from modules.visualize import pose as pose_vis

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-dd",
        "--data_dir",
        required=True,
        type=str,
        help="path of input data",
    )
    parser.add_argument(
        "-vd",
        "--video_dir",
        required=True,
        type=str,
        help="path of input video directory",
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        required=True,
        type=int,
        help="sequential length",
    )
    parser.add_argument(
        "-ts",
        "--th_split",
        required=True,
        type=int,
        help="threshold of split kps",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    return args


def main():
    args = parser()

    seq_len = args.seq_len
    th_split = args.th_split

    # load video paths
    video_dir = args.video_dir
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join(args.data_dir, name)
        os.makedirs(data_dir, exist_ok=True)
        visualise(video_path, data_dir, seq_len, th_split)


def restore_keypoints(pose_data: List[Dict[str, Any]], seq_len: int, th_split: int):
    # sort data by frame_num
    pose_data = sorted(pose_data, key=lambda x: x["frame"])
    # sort data by id
    pose_data = sorted(pose_data, key=lambda x: x["id"])

    # get frame_num and id of first data
    pre_frame_num = pose_data[0]["frame"]
    pre_pid = pose_data[0]["id"]
    pre_kps = pose_data[0]["keypoints"]

    ret_data = []
    seq_data: list = []
    for item in tqdm(pose_data, leave=False):
        # get values
        frame_num = item["frame"]
        pid = item["id"]
        kps = item["keypoints"]

        if np.any(kps[:, 2] < 0.2):
            continue

        if pid != pre_pid:
            if len(seq_data) > seq_len:
                ret_data += _interpolate_kps(seq_data)
            # reset seq_data
            del seq_data
            gc.collect()
            seq_data = []
        else:
            if 1 < frame_num - pre_frame_num and frame_num - pre_frame_num <= th_split:
                # fill brank with nan
                seq_data += [
                    (num, pid, np.full((17, 3), np.nan))
                    for num in range(pre_frame_num + 1, frame_num)
                ]
            elif th_split < frame_num - pre_frame_num:
                if len(seq_data) > seq_len:
                    ret_data += _interpolate_kps(seq_data)
                # reset seq_data
                del seq_data
                gc.collect()
                seq_data = []
            else:
                pass

        # replace nan into low confidence points
        # kps[np.where(kps[:, 2] < 0.2)[0]] = np.nan

        # append keypoints to seq_data
        seq_data.append((frame_num, pid, kps))

        # update frame_num and id
        pre_frame_num = frame_num
        pre_pid = pid
        pre_kps = kps
        del frame_num, pid, kps
        gc.collect()
    else:
        if len(seq_data) > seq_len:
            ret_data += _interpolate_kps(seq_data)
        del seq_data
        gc.collect()

    del pre_frame_num, pre_pid, pre_kps
    del pose_data
    gc.collect()

    return ret_data


def _interpolate_kps(seq_data):
    # collect and kps
    all_kps = np.array([item[2] for item in seq_data])

    # delete last nan
    # print(len(all_kps), all_kps[-1, -1])
    # print(np.where(np.all(np.isnan(all_kps), axis=(1, 2))))
    # all_kps = all_kps[:max(np.where(np.all(~np.isnan(all_kps), axis=(1, 2)))[0]) + 1]
    # print(len(all_kps), all_kps[-1, -1])

    # interpolate and keypoints
    all_kps = all_kps.transpose(1, 0, 2)
    for i in range(len(all_kps)):
        all_kps[i] = _interpolate2d(all_kps[i])
    all_kps = all_kps.transpose(1, 0, 2)

    ret_data = []
    start_frame_num = seq_data[0][0]
    pid = seq_data[0][1]
    for i, kps in enumerate(all_kps):
        ret_data.append(
            {
                "frame": start_frame_num + i,
                "id": pid,
                "keypoints": kps,
            }
        )
    return ret_data


def _interpolate2d(vals):
    ret_vals = np.empty((vals.shape[0], 0))
    for i in range(vals.shape[1]):
        x = np.where(~np.isnan(vals[:, i]))[0]
        y = vals[:, i][~np.isnan(vals[:, i])]
        fitted_curve = interpolate.interp1d(x, y)
        fitted_y = fitted_curve(np.arange(len(vals)))
        fitted_y[np.isnan(vals[:, i])] += np.random.normal(
            0, 5, len(fitted_y[np.isnan(vals[:, i])])
        )

        ret_vals = np.append(ret_vals, fitted_y.reshape(-1, 1), axis=1)
    return ret_vals


def visualise(video_path: str, data_dir: str, seq_len: int, th_split: int):
    # load data
    pose_data_lst = json_handler.load(os.path.join(data_dir, "json", "pose.json"))
    if pose_data_lst is None:
        return

    # create video capture
    print(f"=> loading video from {video_path}.")
    video_capture = Capture(video_path)
    assert (
        video_capture.is_opened
    ), f"{video_path} does not exist or is wrong file type."

    tmp_frame = video_capture.read()[1]
    if tmp_frame is None:
        raise ValueError
    video_capture.set_pos_frame_count(0)

    out_paths = []
    video_num = os.path.basename(video_path).split(".")[0]
    # create video writer for pose estimation results
    out_path = os.path.join(data_dir, f"{video_num}_pose_interpolate.mp4")

    pose_video_writer = Writer(out_path, video_capture.fps, tmp_frame.shape[1::-1])
    out_paths.append(out_path)
    data_lst = restore_keypoints(pose_data_lst, seq_len, th_split)

    print(f"=> writing video into {out_paths}.")
    for frame_num in tqdm(range(video_capture.frame_count), ncols=100):
        frame_num += 1  # frame_num = (1, ...)
        frame = video_capture.read()[1]
        if frame is None:
            raise ValueError

        # write pose estimation video
        frame = pose_vis.write_frame(frame, data_lst, frame_num, False)

        pose_video_writer.write(frame)

    # release memory
    del video_capture
    del pose_video_writer


if __name__ == "__main__":
    main()
