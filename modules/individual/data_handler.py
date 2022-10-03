import os
from logging import Logger
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
from modules.pose import PoseDataHandler
from modules.utils import pickle_handler
from torch.utils.data import DataLoader

from .dataset import IndividualDataset


class IndividualDataHandler:
    @staticmethod
    def create_data_loader(
        data_dirs: List[str],
        config: SimpleNamespace,
        logger: Logger,
        is_test: bool = False,
    ):
        # load pose data
        pose_data_lst: List[List[Dict[str, Any]]] = []
        for data_dir in data_dirs:
            pose_data_lst.append(PoseDataHandler.load(data_dir, logger))

        dataset = IndividualDataset(
            pose_data_lst, config.seq_len, config.th_split, logger
        )
        return DataLoader(dataset, config.batch_size, shuffle=is_test)

    @staticmethod
    def load_generator_data(data_dir, logger: Logger) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "individual_generator.pkl")
        logger.info(f"=> loading individual generator results from {pkl_path}")
        data = pickle_handler.load(pkl_path)
        return data

    @staticmethod
    def save_generator_data(data_dir, data: List[dict], logger: Logger):
        pkl_path = os.path.join(data_dir, "pickle", "individual_generator.pkl")
        logger.info(f"=> saving individual generator results to {pkl_path}")
        pickle_handler.dump(data, pkl_path)

    @staticmethod
    def load_discriminator_data(data_dir, logger: Logger) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "individual_discriminator.pkl")
        logger.info(f"=> loading individual discriminator results from {pkl_path}")
        data = pickle_handler.load(pkl_path)
        return data

    @staticmethod
    def save_discriminator_data(data_dir, data: List[dict], logger: Logger):
        pkl_path = os.path.join(data_dir, "pickle", "individual_discriminator.pkl")
        logger.info(f"=> saving individual discriminator results to {pkl_path}")
        pickle_handler.dump(data, pkl_path)

    @staticmethod
    def get_config(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = SimpleNamespace(**config)
        config.model = SimpleNamespace(**config.model)
        config.model.G = SimpleNamespace(**config.model.G)
        config.model.D = SimpleNamespace(**config.model.D)
        config.optim = SimpleNamespace(**config.optim)

        config.model.G.seq_len = config.seq_len
        config.model.D.seq_len = config.seq_len

        return config
