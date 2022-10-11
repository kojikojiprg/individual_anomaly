import os
from types import SimpleNamespace
from typing import Any, Dict, List

import yaml
from modules.utils import pickle_handler
from pytorch_lightning import LightningDataModule

from .datamodule import IndividualDataModule


class IndividualDataHandler:
    @staticmethod
    def get_config(model_type: str) -> SimpleNamespace:
        config_path = IndividualDataHandler._get_config_path(model_type)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = IndividualDataHandler._get_config_reccursive(config)

        model_names = config.model.__dict__.keys()

        # check included model
        assert "D" in model_names and ("G" in model_names or "E" in model_names)

        # check d_z of egan and autoencoder (not included gan)
        if "G" in model_names and "E" in model_names:
            assert config.model.G.d_z == config.model.E.d_z
        if "D" in model_names and "E" in model_names:
            assert config.model.D.d_z == config.model.E.d_z

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

    @staticmethod
    def create_datamodule(
        data_dir: str, config: SimpleNamespace, stage: str = None
    ) -> LightningDataModule:
        return IndividualDataModule(data_dir, config, stage)

    @staticmethod
    def load_generator_data(data_dir) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "individual_generator.pkl")
        data = pickle_handler.load(pkl_path)
        return data

    @staticmethod
    def save_generator_data(data_dir, data: List[dict]):
        pkl_path = os.path.join(data_dir, "pickle", "individual_generator.pkl")
        pickle_handler.dump(data, pkl_path)

    @staticmethod
    def load_discriminator_data(data_dir) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(data_dir, "pickle", "individual_discriminator.pkl")
        data = pickle_handler.load(pkl_path)
        return data

    @staticmethod
    def save_discriminator_data(data_dir, data: List[dict]):
        pkl_path = os.path.join(data_dir, "pickle", "individual_discriminator.pkl")
        pickle_handler.dump(data, pkl_path)