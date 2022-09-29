import os
from logging import Logger
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
from modules.pose import PoseDataHandler
from modules.utils import pickle_handler
from torch.utils.data import DataLoader

from .dataset import IndividualDataset
from .model import IndividualGAN


class IndividualDataHandler:
    @staticmethod
    def create_data_loader(data_dirs: str, config: SimpleNamespace, logger: Logger):
        # load pose data
        pose_data_lst: List[List[Dict[str, Any]]] = []
        for data_dir in data_dirs:
            pose_data_lst.append(PoseDataHandler.load(data_dir, logger))

        dataset = IndividualDataset(
            pose_data_lst, config.seq_len, config.th_split, logger
        )
        return DataLoader(dataset, config.batch_size, shuffle=True)

    @staticmethod
    def load(data_dir, logger: Logger) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "individual.pkl")

        logger.info(f"=> loading individual activity results from {pkl_path}")
        pkl_data = pickle_handler.load(pkl_path)
        return pkl_data

    @staticmethod
    def save(data_dir, data: List[dict], logger: Logger):
        pkl_path = os.path.join(data_dir, "pickle", "individual.pkl")

        logger.info(f"=> saving individual activity results to {pkl_path}")
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


class IndividualModelFactory:
    @staticmethod
    def create_model(config: SimpleNamespace, device: str):
        return IndividualGAN(config, device)

    @staticmethod
    def load_model(
        checkpoint_dir: str,
        config: SimpleNamespace,
        device: str,
    ) -> IndividualGAN:
        pass

    @staticmethod
    def save_model(
        model: IndividualGAN,
        checkpoint_dir: str,
        config: SimpleNamespace,
        device: str,
    ):
        pass
