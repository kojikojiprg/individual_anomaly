import gc
import os
from glob import glob
from types import SimpleNamespace
from typing import List, Tuple, Union

import yaml

from modules.utils import json_handler
from modules.utils.constants import Stages

from .constants import (
    IndividualDataFormat,
    IndividualDataTypes,
    IndividualModelTypes,
    IndividualPredTypes,
)
from .models.ganomaly_bbox.datamodule import IndividualDataModuleBbox
from .models.ganomaly_kps.datamodule import IndividualDataModuleKps
from .models.role_estimation.datamodule import RoleEstimationDataModule


class IndividualDataHandler:
    @classmethod
    def get_config(
        cls,
        model_type: str,
        seq_len: int,
        data_type: str,
        stage: str = Stages.inference,
    ) -> SimpleNamespace:
        config_path = cls._get_config_path(model_type, seq_len, data_type)
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

        # copy seq_len
        if seq_len is not None:
            assert seq_len == config.dataset.seq_len
        if "G" in model_names:
            config.model.G.seq_len = config.dataset.seq_len
        if "D" in model_names:
            config.model.D.seq_len = config.dataset.seq_len
        if "E" in model_names:
            config.model.E.seq_len = config.dataset.seq_len

        # set batch_size
        if stage == Stages.train:
            config.dataset.batch_size = config.train.batch_size
        else:
            config.dataset.batch_size = config.inference.batch_size

        return config

    @staticmethod
    def _get_config_path(model_type: str, seq_len: int, data_type: str):
        if model_type == IndividualModelTypes.ganomaly:
            return os.path.join(
                "configs",
                "individual",
                f"{model_type.lower()}_{data_type.lower()}_seq{seq_len}.yaml",
            )
        else:
            return os.path.join(
                "configs",
                "individual",
                f"{model_type.lower()}_seq{seq_len}.yaml",
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
        model_type: str,
        data_type: str = IndividualDataTypes.local,
        stage: str = Stages.inference,
        frame_shape: Tuple[int, int] = None,
        annotation_path: str = None,
    ) -> Union[IndividualDataModuleBbox, IndividualDataModuleKps]:
        if model_type == IndividualModelTypes.ganomaly:
            if data_type == IndividualDataTypes.bbox:
                return IndividualDataModuleBbox(data_dir, config, stage, frame_shape)
            else:
                return IndividualDataModuleKps(
                    data_dir, config, data_type, stage, frame_shape
                )
        elif model_type == IndividualModelTypes.role_estimation:
            return RoleEstimationDataModule(
                data_dir, annotation_path, config, stage, frame_shape
            )
        else:
            raise ValueError

    @staticmethod
    def _get_data_path(
        data_dir: str,
        model_type: str,
        data_type: str,
        masking: bool,
        seq_len: int,
        prediction_type: str,
    ):
        str_masked = ""
        if masking and data_type != IndividualDataTypes.bbox:
            str_masked = "_masked"

        str_pred = ""
        if (
            prediction_type == IndividualPredTypes.keypoints
            and data_type != IndividualDataTypes.bbox
        ):
            str_pred = "_kps"

        return os.path.join(
            data_dir,
            "json",
            f"individual_{model_type}{str_masked}_{data_type}_seq{seq_len}{str_pred}.json",
        )

    @classmethod
    def load(
        cls,
        data_dir: str,
        model_type: str,
        data_type: str,
        masking: bool,
        seq_len: int,
        prediction_type: str,
        data_keys: list = None,
    ) -> List[dict]:
        data_path = cls._get_data_path(
            data_dir, model_type, data_type, masking, seq_len, prediction_type
        )
        data = json_handler.load(data_path)
        if data_keys is None:
            return data
        else:
            data_keys = data_keys + [
                IndividualDataFormat.frame_num,
                IndividualDataFormat.id,
            ]
            data_keys + list(set(data_keys))  # get unique

            ret_data = [{k: item[k] for k in data_keys} for item in data]
            del data
            gc.collect()
            return ret_data

    @classmethod
    def save(
        cls,
        data_dir: str,
        data,
        model_type: str,
        data_type: str,
        masking: bool,
        seq_len: int,
        prediction_type: str,
    ):
        data_path = cls._get_data_path(
            data_dir, model_type, data_type, masking, seq_len, prediction_type
        )
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        json_handler.dump(data, data_path)

    @staticmethod
    def get_frame_shape(data_dir: str):
        # for 0-1 scaling keypoints
        path = glob(os.path.join(data_dir, "*", "json", "frame_shape.json"))[0]
        frame_shape = json_handler.load(path)
        return frame_shape
