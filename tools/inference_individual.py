import argparse
import os
import sys
import warnings

sys.path.append(".")
from modules.individual import IndividualActivityRecognition
from modules.individual.constants import IndividualDataTypes
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
    parser.add_argument(
        "-sl",
        "--seq_len",
        required=True,
        type=int,
        help="sequential length",
    )

    # options
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        default="ganomaly",
        help="'ganomaly' only",
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

    model_type = args.model_type
    data_type = args.data_type
    seq_len = args.seq_len
    checkpoint_path = os.path.join(
        "models",
        "individual",
        model_type,
        f"{model_type}_masked_{data_type}_seq{seq_len}_last.ckpt",
    )
    iar = IndividualActivityRecognition(
        model_type,
        logger,
        checkpoint_path,
        seq_len=seq_len,
        data_type=data_type,
        stage="inference",
    )
    iar.inference(args.data_dir, [args.gpu], args.video_dir)


if __name__ == "__main__":
    main()
