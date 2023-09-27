import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from modules.utils.video import Capture


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

    return parser.parse_args()


def main():
    args = parser()

    # load video paths
    video_dir = args.video_dir
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    out_paths = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        out_dir = os.path.join(args.output_dir, name, "bin")
        os.makedirs(out_dir, exist_ok=True)
        out_paths.append(os.path.join(out_dir, "flow.npy"))

    for video_path, out_path in zip(video_paths, out_paths):
        print(f"=> processing {video_path}")
        cap = Capture(video_path)
        w, h = cap.size

        flows = np.empty((0, h, w, 2))
        flows = np.append(flows, np.zeros((1, h, w, 2), np.float32), axis=0)
        pre_gray = None
        for i in tqdm(range(cap.frame_count)):
            frame = cap.read()[1]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if i == 0:
                pre_gray = gray
                continue

            flow = cv2.calcOpticalFlowFarneback(pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows = np.append(flows, flow.reshape(1, h, w, 2), axis=0)

            pre_gray = gray

        np.save(out_path, flows)
        del cap


if __name__ == "__main__":
    main()
