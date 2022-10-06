import os
from logging import Logger
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
from modules.pose import PoseDataHandler
from modules.utils import pickle_handler, video
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

        # get frame size
        data_dir = data_dirs[0]
        video_path = os.path.join(data_dir, "pose.mp4")
        cap = video.Capture(video_path)
        frame_shape = cap.size
        del cap

        # create dataset
        dataset = IndividualDataset(
            pose_data_lst, config.seq_len, config.th_split, frame_shape, logger
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
    def get_config(model_type: str):
        config_path = IndividualDataHandler._get_config_path(model_type)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = IndividualDataHandler._get_config_reccursive(config)

        model_names = config.model.__dict__.keys()

        # check included model
        assert "D" in model_names and ("G" in model_names or "E" in model_names)

        # check d_z and d_model
        if "G" in model_names and "D" in model_names:
            assert config.model.G.d_model == config.model.D.d_model
        if "G" in model_names and "E" in model_names:
            assert config.model.G.d_z == config.model.E.d_z
        if "D" in model_names and "E" in model_names:
            assert config.model.D.d_model == config.model.E.d_model

        # set same seq_len
        if "G" in model_names:
            config.model.G.seq_len = config.seq_len
        if "D" in model_names:
            config.model.D.seq_len = config.seq_len
        if "E" in model_names:
            config.model.E.seq_len = config.seq_len

        return config

    @staticmethod
    def _get_config_path(model_type: str):
        return os.path.join("configs", "individual", f"{model_type.lower()}.yaml")

    @staticmethod
    def _get_config_reccursive(config: dict):
        new_config = SimpleNamespace(**config)
        for name, values in new_config.__dict__.items():
            if type(values) == dict:
                new_config.__setattr__(
                    name, IndividualDataHandler._get_config_reccursive(values)
                )
            else:
                continue
        return new_config
