import os
from glob import glob
from types import SimpleNamespace
from typing import List, Tuple

import yaml

from modules.utils import pickle_handler
from modules.utils.constants import Stages
from modules.utils.video import Capture

from .constants import IndividualDataFormat, IndividualDataTypes
from .datamodule import IndividualDataModule


class IndividualDataHandler:
    @classmethod
    def get_config(
        cls,
        model_type: str,
        data_type: str,
        stage: str = Stages.inference,
        seq_len: int = None,
    ) -> SimpleNamespace:
        config_path = cls._get_config_path(model_type, data_type, seq_len)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = cls._get_config_reccursive(config)

        model_names = config.model.__dict__.keys()

        # check included model
        assert "D" in model_names and ("G" in model_names or "E" in model_names)

        # check d_z of egan and autoencoder (not included gan)
        if "G" in model_names and "E" in model_names:
            assert config.model.G.d_z == config.model.E.d_z
        if "D" in model_names and "E" in model_names:
            assert config.model.D.d_z == config.model.E.d_z

        # select n_kps by data_type
        if data_type == IndividualDataTypes.both:
            n_kps = 34  # both
        elif data_type == IndividualDataTypes.bbox:
            n_kps = None
        else:
            n_kps = 17

        # set same seq_len, n_kps
        if seq_len is not None:
            assert seq_len == config.dataset.seq_len
        config.n_kps = n_kps
        if "G" in model_names:
            config.model.G.seq_len = config.dataset.seq_len
            config.model.G.n_kps = n_kps
        if "D" in model_names:
            config.model.D.seq_len = config.dataset.seq_len
            config.model.D.n_kps = n_kps
        if "E" in model_names:
            config.model.E.seq_len = config.dataset.seq_len
            config.model.E.n_kps = n_kps

        # set batch_size
        if stage == Stages.train:
            config.dataset.batch_size = config.train.batch_size
        else:
            config.dataset.batch_size = config.inference.batch_size

        return config

    @staticmethod
    def _get_config_path(model_type: str, data_type: str, seq_len: int = None):
        if seq_len is None:
            return os.path.join(
                "configs",
                "individual",
                f"{model_type.lower()}_{data_type.lower()}.yaml",
            )
        else:
            return os.path.join(
                "configs",
                "individual",
                f"{model_type.lower()}_{data_type.lower()}_seq{seq_len}.yaml",
            )

    @classmethod
    def _get_config_reccursive(cls, config: dict):
        new_config = SimpleNamespace(**config)
        for name, values in new_config.__dict__.items():
            if type(values) == dict:
                new_config.__setattr__(name, cls._get_config_reccursive(values))
            else:
                continue
        return new_config

    @staticmethod
    def create_datamodule(
        data_dir: str,
        config: SimpleNamespace,
        data_type: str = IndividualDataTypes.local,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
    ) -> IndividualDataModule:
        return IndividualDataModule(data_dir, config, data_type, stage, frame_shape)

    @staticmethod
    def _get_data_path(
        data_dir: str, model_type: str, data_type: str, seq_len: int = None
    ):
        if seq_len is None:
            return os.path.join(
                data_dir, "pickle", f"individual_{model_type}_masked_{data_type}.pkl"
            )
        else:
            return os.path.join(
                data_dir,
                "pickle",
                f"individual_{model_type}_masked_{data_type}_seq{seq_len}.pkl",
            )

    @classmethod
    def load(
        cls,
        data_dir: str,
        model_type: str,
        data_type: str,
        data_keys: list = None,
        seq_len: int = None,
    ) -> List[dict]:
        pkl_path = cls._get_data_path(data_dir, model_type, data_type, seq_len)
        data = pickle_handler.load(pkl_path)
        if data_keys is None:
            return data
        else:
            data_keys = set(
                data_keys + [IndividualDataFormat.frame_num, IndividualDataFormat.id]
            )
            ret_data = [{k: item[k] for k in data_keys} for item in data]
            del data
            return ret_data

    @classmethod
    def save(
        cls, data_dir: str, data, model_type: str, data_type: str, seq_len: int = None
    ):
        pkl_path = cls._get_data_path(data_dir, model_type, data_type, seq_len)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        pickle_handler.dump(data, pkl_path)

    @staticmethod
    def get_frame_shape(video_dir: str):
        # for 0-1 scaling keypoints
        video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
        cap = Capture(video_paths[0])
        frame_shape = cap.size
        del cap
        return frame_shape
