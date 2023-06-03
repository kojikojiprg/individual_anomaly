import argparse
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
        "-mv",
        "--model_version",
        type=int,
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
        "-an",
        "--annotation_path",
        type=str,
        default=None,
        help="'role_estimation': annotation file path",
    )

    args = parser.parse_args()

    # delete last slash
    args.data_dir = args.data_dir[:-1] if args.data_dir[-1] == "/" else args.data_dir

    return args


def main():
    args = parser()

    data_type = args.data_type
    seq_len = args.seq_len
    iar = IndividualActivityRecognition(
        "role_estimation",
        seq_len,
        data_type=data_type,
        stage="inference",
        model_version=args.model_version,
    )
    iar.inference(args.data_dir, [args.gpu], annotation_path=args.annotation_path)


if __name__ == "__main__":
    main()
