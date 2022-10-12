import argparse
import os
import sys
import warnings

sys.path.append(".")
from modules.individual import (
    IndividualActivityRecognition,
    IndividualDataTypes,
    IndividualModelTypes,
)
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
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="both",
        help="Input data type. Selected by 'abs', 'rel' or 'both', by defualt is 'both'.",
    )
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="gpu ids")
    args = parser.parse_args()

    assert IndividualModelTypes.includes(args.model_type)
    assert IndividualDataTypes.includes(args.data_type)

    return args


def main():
    args = parser()

    data_dir = os.path.join("data", args.dataset)

    iar = IndividualActivityRecognition(
        args.model_type,
        data_dir,
        args.gpus,
        logger,
        data_type=args.data_type,
        stage="train",
    )
    iar.train()


if __name__ == "__main__":
    main()
