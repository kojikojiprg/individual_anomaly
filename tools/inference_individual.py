import argparse
import os
import sys
import warnings

sys.path.append(".")
from modules.individual import IndividualActivityRecognition

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
        "-mv",
        "--model_version",
        type=str,
        default=None,
        help="model version",
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        type=str,
        default="local",
        help="Input data type. Selected by 'global', 'local', 'local+bbox' or 'both', by defualt is 'local'.",
    )
    parser.add_argument(
        "-msk",
        "--masking",
        default=False,
        action="store_true",
        help="Masking low confidence score keypoints",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    args.model_type = args.model_type.lower()

    return args


def main():
    args = parser()

    model_type = args.model_type
    if args.model_version is not None:
        model_version = f"-v{args.model_version}"
    else:
        model_version = ""
    data_type = args.data_type
    masking = args.masking
    seq_len = args.seq_len
    checkpoint_path = os.path.join(
        "models",
        "individual",
        model_type,
        f"{model_type}_masked_{data_type}_seq{seq_len}_last{model_version}.ckpt",
    )
    iar = IndividualActivityRecognition(
        model_type,
        seq_len,
        checkpoint_path,
        data_type=data_type,
        masking=masking,
        stage="inference",
    )
    iar.inference(args.data_dir, [args.gpu])


if __name__ == "__main__":
    main()
