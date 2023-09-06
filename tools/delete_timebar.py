import argparse
import os
import sys
from glob import glob

from numpy.typing import NDArray
from tqdm import tqdm

sys.path.append("src")
from utility.logger import logger
from utility.video import Capture, Writer


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rn",
        "--room_num",
        required=True,
        type=str,
        help="operating room number",
    )
    parser.add_argument(
        "-s",
        "--surgery_num",
        required=True,
        type=str,
        help="surgery number of each room",
    )
    parser.add_argument("-tl", "--timebar_height", type=int, default=20)
    parser.add_argument("-ex", "--expand_name", type=str, default="")

    return parser.parse_args()


def _delete_time_bar(frame: NDArray, delete_hight: int = 20):
    return frame[delete_hight:]


def main():
    args = _setup_parser()
    video_files = sorted(
        glob(
            os.path.join(
                "video", args.room_num, args.surgery_num, args.expand_name, "*"
            )
        )
    )
    logger.info(f"=> {video_files}")

    for video_file in video_files:
        cap = Capture(video_file)

        write_video_file = video_file.replace(".mp4", "_new.mp4")
        size = (cap.size[0], cap.size[1] - args.timebar_height)
        wrt = Writer(write_video_file, cap.fps, size)

        logger.info(f"=> writing {video_file} to {write_video_file}")
        for _ in tqdm(range(cap.frame_count)):
            _, frame = cap.read()

            frame = _delete_time_bar(frame, args.timebar_height)
            wrt.write(frame)

        del cap, wrt
        os.remove(video_file)
        os.rename(write_video_file, video_file)
        logger.info(f"=> rename {video_file} to {write_video_file}")


if __name__ == "__main__":
    main()
