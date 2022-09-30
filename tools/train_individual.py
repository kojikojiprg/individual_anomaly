import argparse
import os
import sys
import warnings
from glob import glob

import torch

sys.path.append(".")
from modules.individual import IndividualDataHandler, IndividualModelFactory
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

    # options
    parser.add_argument(
        "-ex", "--expand_name", type=str, default="", help="'passing' or 'attention'"
    )
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="configs/individual/individual.yaml"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    args = parser.parse_args()

    return args


def main():
    args = parser()

    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load video paths
    data_dir = os.path.join("data", args.room_num, args.surgery_num, args.expand_name)
    dirs = sorted(glob(os.path.join(data_dir, "*")))
    logger.info(f"=> data directories:\n{dirs}")

    # load config
    config = IndividualDataHandler.get_config(args.cfg_path)

    # creating dataloader
    dataloader = IndividualDataHandler.create_data_loader(dirs, config, logger)

    checkpoint_dir = os.path.join("models", "individual")
    # load model
    if os.path.exists(checkpoint_dir):
        model = IndividualModelFactory.load_model(
            checkpoint_dir, config, device, logger
        )
    else:
        model = IndividualModelFactory.create_model(config, device, logger)

    # train
    model.train(dataloader)

    # save params
    IndividualModelFactory.save_model(model, checkpoint_dir)


if __name__ == "__main__":
    main()
