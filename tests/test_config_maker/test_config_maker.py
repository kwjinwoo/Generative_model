import pytest

from gen_ai.configs import (
    ConfigMaker,
    GenAIConfig,
    check_config_path,
    check_extension,
    check_path,
    get_model_type,
    load_yaml,
)
from gen_ai.configs.data_configs import DataConfig
from gen_ai.configs.model_configs import AutoregressiveConfig
from gen_ai.configs.train_configs import TrainConfig


@pytest.mark.parametrize(
    "path, valid",
    [
        ("tests/test_config_maker/test_config_maker/autoregressive_config.yaml", True),
        ("wrong_path", False),
    ],
)
def test_check_path(path, valid):
    if not valid:
        with pytest.raises(FileNotFoundError):
            check_path(path)
    else:
        check_path(path)


@pytest.mark.parametrize(
    "path, valid",
    [
        ("tests/test_config_maker/test_config_maker/autoregressive_config.yaml", True),
        ("tests/test_config_maker/test_config_maker/autoregressive_config.yml", False),
    ],
)
def test_check_extension(path, valid):
    if not valid:
        with pytest.raises(ValueError):
            check_extension(path)
    else:
        check_extension(path)


@pytest.mark.parametrize(
    "path, path_valid, ext_valid",
    [
        ("tests/test_config_maker/test_config_maker/autoregressive_config.yaml", True, True),
        ("wrong_path", False, False),
        ("tests/test_config_maker/test_config_maker/wrong.yml", True, False),
    ],
)
def test_check_config_path(path, path_valid, ext_valid):
    if not path_valid:
        with pytest.raises(FileNotFoundError):
            check_config_path(path)
    elif not ext_valid:
        with pytest.raises(ValueError):
            check_config_path(path)
    else:
        check_config_path(path)


def test_load_yaml(datadir):
    path = datadir / "autoregressive_config.yaml"
    config = load_yaml(path)
    assert isinstance(config, dict)

    assert "model" in config
    assert "data" in config
    assert "train" in config

    assert config["model"]["model_type"] == "autoregressive"
    assert config["model"]["img_channel"] == 3
    assert config["model"]["num_channels"] == 128
    assert config["model"]["num_layers"] == 3

    assert config["data"]["data_type"] == "mnist"

    assert config["train"]["num_epochs"] == 10


@pytest.mark.parametrize("config_file, answer_model_type", [("autoregressive_config_file", "autoregressive")])
def test_get_model_type(config_file, answer_model_type, request):
    config = request.getfixturevalue(config_file)
    model_type = get_model_type(config)
    assert model_type == answer_model_type


@pytest.mark.parametrize("config_file, file_name", (("autoregressive_config_file", "autoregressive_config.yaml"),))
def test_config_maker(config_file, file_name, request, datadir):
    config_file = request.getfixturevalue(config_file)
    config_maker = ConfigMaker(str(datadir / file_name))
    config = config_maker.make_config()

    assert isinstance(config, GenAIConfig)
    assert isinstance(config.model_config, AutoregressiveConfig)
    assert isinstance(config.data_config, DataConfig)
    assert isinstance(config.train_config, TrainConfig)

    assert config.model_config.model_type == config_file["model"]["model_type"]
    assert config.model_config.img_channel == config_file["model"]["img_channel"]
    assert config.model_config.num_channels == config_file["model"]["num_channels"]

    assert config.data_config.data_type == config_file["data"]["data_type"]
    assert config.data_config.batch_size == config_file["data"]["batch_size"]

    assert config.train_config.num_epochs == config_file["train"]["num_epochs"]
