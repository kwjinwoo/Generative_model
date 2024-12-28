import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

from gen_ai.configs.data_configs import DataConfig
from gen_ai.configs.model_configs import AutoregressiveConfig, ModelConfig
from gen_ai.configs.train_configs import TrainConfig
from gen_ai.enums import ModelType


@dataclass
class GenAIConfig:
    model_config: ModelConfig
    data_config: DataConfig
    train_config: TrainConfig


YAML = Dict[str, Any]


def check_path(path: str) -> None:
    """check if the path is valid file path.

    Args:
        path (str): path to check.

    Raises:
        FileNotFoundError: if the path is not valid file path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file is not Found. {path}")


def check_extension(path: str) -> None:
    """check if the path is yaml file.

    Args:
        path (str): check if the path is yaml file.

    Raises:
        ValueError: if the path is not yaml file.
    """
    if not path.endswith(".yaml"):
        raise ValueError("Config file must be YAML format.")


def check_config_path(config_path: str) -> None:
    """check if the config path is valid.

    Args:
        config_path (str): config path to check.
    """
    check_path(config_path)
    check_extension(config_path)


def load_yaml(path: str) -> YAML:
    """load yaml file.

    Args:
        path (str): path to yaml file.

    Returns:
        YAML: loaded yaml file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_type(config_file: YAML) -> str:
    """get model type from config file.

    Args:
        config_file (YAML): config file.

    Raises:
        KeyError: if model type is not defined in config file.

    Returns:
        str: model type.
    """
    model_type = config_file["model"].get("model_type")
    if model_type is None:
        raise KeyError("Model Type is not defined in Config File.")
    return model_type.lower()


class ConfigMaker:
    """Config Maker Class.
    it makes config object from config file.
    """

    def __init__(self, config_path: str) -> None:
        """Config Maker Constructor.

        Args:
            config_path (str): path to config file.
        """
        check_config_path(config_path)
        self.config_file = load_yaml(config_path)

    def make_config(self) -> GenAIConfig:
        """make GenAIConfig object from config file.

        Returns:
            GenAIConfig: GenAIConfig object.
        """
        model_config = self._get_model_config()
        data_config = DataConfig(**self.config_file["data"])
        train_config = TrainConfig(**self.config_file["train"])
        return GenAIConfig(model_config, data_config, train_config)

    def _get_model_config(self) -> ModelConfig:
        """get model config object from config file.

        Raises:
            ValueError: if model type is not supported.

        Returns:
            ModelConfig: model config object.
        """
        model_type = get_model_type(self.config_file)

        if model_type == ModelType.autoregressive:
            return AutoregressiveConfig(**self.config_file["model"])
        else:
            raise ValueError(f"Model Type {model_type} is not supported.")


__all__ = ["GenAIConfig", "ConfigMaker"]
