import argparse
import os
import sys
import warnings
from glob import glob

import torch

sys.path.append(".")
from modules.individual import IndividualActivityRecognition
from modules.utils.logger import logger

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
    parser.add_argument(
        "-mt",
        "--model_type",
        required=True,
        type=str,
        default="GAN",
        help="'GAN(gan)', 'EGAN(egan)' or 'Autoencoder(autoencoder)'",
    )

    # options
    parser.add_argument(
        "-ex", "--expand_name", type=str, default="", help="'passing' or 'attention'"
    )
    parser.add_argument("-nm", "--load_model", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    args = parser.parse_args()

    return args


def main():
    args = parser()

    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load video paths
    data_dir = os.path.join("data", args.room_num, args.surgery_num, args.expand_name)
    data_dirs = sorted(glob(os.path.join(data_dir, "*")))
    logger.info(f"=> data directories:\n{data_dirs}")

    iar = IndividualActivityRecognition(args.model_type, device, logger)
    iar.train(data_dirs, args.load_model)


if __name__ == "__main__":
    main()
