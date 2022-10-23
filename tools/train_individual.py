import argparse
import sys
import warnings

sys.path.append(".")
from modules.individual import IndividualActivityRecognition
from modules.utils.logger import logger

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-dir",
        "--data_dir",
        required=True,
        type=str,
        default="data/dataset/01/train",
        help="path of input data",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        required=True,
        type=str,
        default="GAN",
        help="'GAN(gan)', 'EGAN(egan)' or 'Autoencoder(autoencoder)'",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        required=True,
        type=str,
        default="local",
        help="Input data type. Selected by 'global', 'global_bbox', 'local' or 'both', by defualt is 'local'.",
    )

    # options
    parser.add_argument(
        "-g", "--gpus", type=int, nargs="*", default=None, help="gpu ids"
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    return args


def main():
    args = parser()

    iar = IndividualActivityRecognition(
        args.model_type,
        logger,
        data_type=args.data_type,
        stage="train",
    )
    iar.train(args.data_dir, args.gpus)


if __name__ == "__main__":
    main()
