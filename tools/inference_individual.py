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
        default="egan",
        help="'egan' or 'ganomaly'",
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
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    return args


def main():
    args = parser()

    model_type = args.model_type.lower()
    data_type = args.data_type
    checkpoint_path = os.path.join(
        "models",
        "individual",
        model_type,
        f"{model_type}_last-{data_type}.ckpt",
    )
    iar = IndividualActivityRecognition(
        model_type,
        logger,
        checkpoint_path,
        data_type=data_type,
        stage="inference",
    )
    iar.inference(args.data_dir, args.gpu_id)


if __name__ == "__main__":
    main()
