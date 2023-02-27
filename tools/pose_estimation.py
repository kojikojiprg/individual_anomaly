import argparse
import os
import sys
import warnings
from glob import glob

import torch

sys.path.append(".")
from modules.pose import PoseEstimation
from modules.visualize import Visualizer

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-vd",
        "--video_dir",
        required=True,
        type=str,
        help="path of input video directory",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        required=True,
        type=str,
        help="path of output data directory",
    )

    # options
    parser.add_argument("-c", "--cfg_path", type=str, default="configs/pose/pose.yaml")
    parser.add_argument("-dg", "--det_gpu", type=int, default=0, help="gpu number of detector")
    parser.add_argument("-tg", "--trk_gpu", type=int, default=0, help="gpu number of tracker")
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

    torch.cuda.set_device(args.det_gpu)

    # load video paths
    video_dir = args.video_dir
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join(args.output_dir, name)
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    # load model
    pe = PoseEstimation(args.cfg_path, args.det_gpu, args.trk_gpu)

    if args.video:
        vis = Visualizer(args)

    for video_path, data_dir in zip(video_paths, data_dirs):
        print(f"=> processing {video_path}")
        pe.inference(video_path, data_dir)

        if args.video:
            vis.visualise(video_path, data_dir)


if __name__ == "__main__":
    main()
