import os
from glob import glob
from types import SimpleNamespace
from typing import Dict, List

import yaml
from modules.utils import pickle_handler
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm

from .datamodule import IndividualDataModule


class IndividualDataHandler:
    @staticmethod
    def get_config(model_type: str, stage: str = None) -> SimpleNamespace:
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
            config.model.G.seq_len = config.dataset.seq_len
        if "D" in model_names:
            config.model.D.seq_len = config.dataset.seq_len
        if "E" in model_names:
            config.model.E.seq_len = config.dataset.seq_len

        # set batch_size
        if stage == "train":
            config.dataset.batch_size = config.train.batch_size
        else:
            config.dataset.batch_size = config.inference.batch_size

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
    def _load_data_each_stage(model_type, data_dir, stage):
        data_dirs = sorted(glob(os.path.join(data_dir, stage, "*")))
        data = {}
        for path in tqdm(data_dirs, desc=stage):
            video_num = os.path.basename(path)
            pkl_path = os.path.join(path, "pickle", f"individual_{model_type}.pkl")
            data[video_num] = pickle_handler.load(pkl_path)
        return data

    @classmethod
    def load_data(
        cls, model_type: str, data_dir: str
    ) -> Dict[str, Dict[str, List[dict]]]:
        train_data = cls._load_data_each_stage(model_type, data_dir, "train")
        test_data = cls._load_data_each_stage(model_type, data_dir, "test")

        return {"train": train_data, "test": test_data}

    @staticmethod
    def save_data(
        model_type: str, data_dir: str, data: Dict[str, Dict[str, List[dict]]]
    ):
        for stage, stage_results in data.items():
            for video_num, result in tqdm(stage_results.items(), desc=stage):
                pkl_path = os.path.join(
                    data_dir, stage, video_num, "pickle", f"individual_{model_type}.pkl"
                )
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
                pickle_handler.dump(result, pkl_path)
