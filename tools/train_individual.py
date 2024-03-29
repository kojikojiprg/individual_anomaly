#!/usr/bin/env python
import argparse
import sys
import warnings

sys.path.append(".")
from src import IndividualAnomalyEstimation

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-dd",
        "--data_dir",
        required=True,
        type=str,
        help="path of input data",
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        required=True,
        type=int,
        help="sequential length",
    )

    # options
    parser.add_argument(
        "-g", "--gpus", type=int, nargs="*", default=None, help="gpu ids"
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="local",
        help="'ganomaly': Input data type. Selected by 'global', 'local', or 'both', by defualt is 'local'.",
    )
    parser.add_argument(
        "-msk",
        "--masking",
        default=False,
        action="store_true",
        help="'ganomaly': Masking low confidence score keypoints",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    return args


def main():
    args = parser()

    iar = IndividualAnomalyEstimation(
        args.seq_len,
        data_type=args.data_type,
        masking=args.masking,
        stage="train",
    )
    iar.train(args.data_dir, args.gpus)


if __name__ == "__main__":
    main()
