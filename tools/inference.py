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
        "-rn", "--room_num", required=True, type=str, help="room number"
    )
    parser.add_argument(
        "-s",
        "--surgery_num",
        required=True,
        type=str,
        help="surgery number of each room",
    )

    # options
    parser.add_argument(
        "-ex", "--expand_name", type=str, default="", help="'passing' or 'attention'"
    )
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/surgery_config.yaml"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    parser.add_argument(
        "-k",
        "--keypoints",
        default=False,
        action="store_true",
        help="with keypoints extraction",
    )
    parser.add_argument(
        "-i",
        "--individual",
        default=False,
        action="store_true",
        help="with idividual recognition",
    )
    parser.add_argument(
        "-g",
        "--group",
        default=False,
        action="store_true",
        help="without group recognition",
    )
    parser.add_argument(
        "-v", "--video", default=False, action="store_true", help="with writing video"
    )

    return parser.parse_args()


def main():
    args = parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # load video paths
    if args.expand_name != "":
        video_dir = os.path.join(
            "video", args.room_num, args.surgery_num, args.expand_name
        )
    else:
        video_dir = os.path.join(
            "/raid6/surgery-video", args.room_num, args.surgery_num, args.expand_name
        )
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    if args.keypoints:
        pose_model = PoseEstimation(args.cfg_path, device, logger)

    vis = Visualizer(logger)

    for video_path, data_dir in zip(video_paths, data_dirs):
        logger.info(f"=> processing {video_path}")

        if args.keypoints:
            pose_model.predict(video_path, data_dir)

        if args.video:
            vis.visualise(video_path, data_dir)


if __name__ == "__main__":
    main()
