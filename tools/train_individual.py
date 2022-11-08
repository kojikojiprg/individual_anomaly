import argparse
import sys
import warnings

sys.path.append(".")
from modules.individual import IndividualActivityRecognition, IndividualDataTypes
from modules.utils.logger import logger

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

    # options
    parser.add_argument(
        "-g", "--gpus", type=int, nargs="*", default=None, help="gpu ids"
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="egan",
        help="'egan' or 'ganomaly'",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="local",
        help="Input data type. Selected by 'global', 'local', 'local+bbox' or 'both', by defualt is 'local'.",
    )
    parser.add_argument(
        "-vd",
        "--video_dir",
        type=str,
        default=None,
        help="path of input video directory",
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        type=int,
        default=None,
        help="sequential length",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    args.model_type = args.model_type.lower()

    if args.data_type in [IndividualDataTypes.global_, IndividualDataTypes.both]:
        assert (
            args.video_dir is not None
        ), f"Input video frame shape is required for data_type:{args.data_type}"

    return args


def main():
    args = parser()

    iar = IndividualActivityRecognition(
        args.model_type,
        logger,
        data_type=args.data_type,
        stage="train",
    )
    iar.train(args.data_dir, args.gpus, args.video_dir)


if __name__ == "__main__":
    main()
