import argparse
import os
import sys
import warnings
from glob import glob
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

sys.path.append(".")
from modules.individual import (
    IndividualActivityRecognition,
    IndividualDataHandler,
    IndividualPredTypes,
)
from modules.pose import PoseDataHandler
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

    # options
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="ganomaly",
        help="'ganomaly' only",
    )
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        default=None,
        help="model version",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="local",
        help="Input data type. Selected by 'global', 'local', 'local+bbox' or 'both', by defualt is 'local'.",
    )
    parser.add_argument(
        "-msk",
        "--masking",
        default=False,
        action="store_true",
        help="Masking low confidence score keypoints",
    )
    parser.add_argument(
        "-p",
        "--predict",
        default=False,
        action="store_true",
        help="Do prediction",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    args.model_type = args.model_type.lower()

    return args


def main():
    args = parser()

    model_type = args.model_type
    if args.model_version is not None:
        model_version = f"-v{args.model_version}"
    else:
        model_version = ""
    data_type = args.data_type
    masking = args.masking
    seq_len = args.seq_len
    checkpoint_path = os.path.join(
        "models",
        "individual",
        model_type,
        f"{model_type}_masked_{data_type}_seq{seq_len}_last{model_version}.ckpt",
    )
    if args.predict:
        iar = IndividualActivityRecognition(
            model_type,
            seq_len,
            checkpoint_path,
            data_type=data_type,
            masking=masking,
            stage="inference",
            prediction_type=IndividualPredTypes.keypoints,
        )
        iar.inference(args.data_dir, [args.gpu])

    # load video paths
    video_dir = args.video_dir
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join(args.data_dir, name)
        results = IndividualDataHandler.load(
            data_dir, model_type, data_type, masking, seq_len, "keypoints"
        )
        os.makedirs(data_dir, exist_ok=True)
        visualise(video_path, data_dir, results)


def restore_keypoints(
    pose_data_lst: List[Dict[str, Any]], ind_data_lst: List[Dict[str, Any]]
):
    ret_data = []
    for ind_data in tqdm(ind_data_lst, desc="restore", ncols=100):
        frame_num_ind = ind_data["frame"]
        id_ind = ind_data["id"]
        for pose_data in pose_data_lst:
            frame_num_pose = pose_data["frame"]
            id_pose = pose_data["id"]
            if int(frame_num_pose) == int(frame_num_ind) and int(id_pose) == int(
                id_ind
            ):
                bbox = pose_data["bbox"]
                kps = np.array(ind_data["keypoints_fake"])
                org = bbox[:2]
                wh = bbox[2:4] - bbox[:2]
                kps *= np.repeat([wh], 17, axis=0).reshape(-1, 17, 2)
                kps = kps[:, :, :2] + np.repeat([org], 17, axis=0).reshape(-1, 17, 2)
                kps = kps.astype(np.float32)
                ret_data.append(
                    {
                        "frame": frame_num_ind,
                        "id": id_ind,
                        "keypoints": kps[-1],
                    }
                )
                break
    return ret_data


def visualise(video_path: str, data_dir: str, results: List[Dict[str, Any]]):
    # load data
    pose_data_lst = PoseDataHandler.load(data_dir)
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
    out_path = os.path.join(data_dir, f"{video_num}_pose_fakekps.mp4")

    pose_video_writer = Writer(out_path, video_capture.fps, tmp_frame.shape[1::-1])
    out_paths.append(out_path)
    data_lst = restore_keypoints(pose_data_lst, results)

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
