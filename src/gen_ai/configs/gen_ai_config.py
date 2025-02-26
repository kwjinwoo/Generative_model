import os
from typing import Any

import yaml

from gen_ai.enums import ModelType

YAML = dict[str, Any]


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


class GenAIConfig:
    def __init__(self, config_path: str) -> None:
        """GenAIConfig Constructor.

        Args:
            config_path (str): path to config file.
        """
        check_config_path(config_path)
        self.config_file = load_yaml(config_path)

    @property
    def model_type(self) -> str:
        """Get model type from config file.

        Raises:
            KeyError: if model type is not defined in config file.

        Returns:
            str: model type.
        """
        model_type = self.config_file["model"].get("model_type")
        if model_type is None:
            raise KeyError("Model Type is not defined in Config File.")

        if model_type.lower() not in ModelType.__members__:
            raise ValueError(f"Model Type {model_type} is not supported.")
        return model_type.lower()

    @property
    def module_config(self) -> YAML:
        """Get module config from config file.
        It used for creating pixelcnn module.

        Returns:
            YAML: module config.
        """
        return self.config_file["module_config"]

    @property
    def data_config(self) -> YAML:
        """Get data config from config file.

        Returns:
            YAML: data config.
        """
        return self.config_file["data"]

    @property
    def train_config(self) -> YAML:
        """Get train config from config file.

        Returns:
            YAML: train config.
        """
        return self.config_file["train"]
