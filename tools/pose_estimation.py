import argparse
import os
import sys
import warnings
from glob import glob

import torch

sys.path.append(".")
from modules.pose import PoseEstimation
from modules.utils.logger import logger
from modules.visualize import Visualizer

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-dir", "--data_dir", required=True, type=str, help="path of input data"
    )

    # options
    parser.add_argument("-c", "--cfg_path", type=str, default="configs/pose/pose.yaml")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    parser.add_argument(
        "-v", "--video", default=False, action="store_true", help="with writing video"
    )
    parser.add_argument(
        "-nb",
        "--video_no_background",
        default=False,
        action="store_true",
        help="with writing video in no background",
    )

    args = parser.parse_args()

    if args.video_no_background:
        args.video = True

    return args


def main():
    args = parser()

    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load video paths
    video_dir = os.path.join("/raid6/surgery-video", args.data_dir)
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    logger.info(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join(
            "data", args.room_num, args.surgery_num, args.expand_name, name
        )
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    # load model
    pe = PoseEstimation(args.cfg_path, device, logger)

    if args.video:
        vis = Visualizer(args, logger)

    for video_path, data_dir in zip(video_paths, data_dirs):
        logger.info(f"=> processing {video_path}")
        pe.inference(video_path, data_dir)

        if args.video:
            vis.visualise(video_path, data_dir)


if __name__ == "__main__":
    main()
