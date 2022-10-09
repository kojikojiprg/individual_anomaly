import argparse
import os
import sys
import warnings

sys.path.append(".")
from modules.individual import IndividualActivityRecognition
from modules.utils.logger import logger

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument("-d", "--dataset", required=True, type=str, help="dataset name")
    parser.add_argument(
        "-mt",
        "--model_type",
        required=True,
        type=str,
        default="GAN",
        help="'GAN(gan)', 'EGAN(egan)' or 'Autoencoder(autoencoder)'",
    )

    # options
    parser.add_argument("--gpus", type=int, nargs="*", default=0, help="gpu number")
    args = parser.parse_args()

    return args


def main():
    args = parser()

    data_dir = os.path.join("data", args.dataset)
    logger.info(f"=> data directorie: {data_dir}")

    iar = IndividualActivityRecognition(
        args.model_type, data_dir, args.gpus, logger, "train"
    )
    iar.train()


if __name__ == "__main__":
    main()
