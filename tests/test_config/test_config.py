import pytest

from gen_ai.configs.gen_ai_config import check_config_path, check_extension, check_path, load_yaml


@pytest.mark.parametrize(
    "file_name, valid",
    [
        ("autoregressive_config.yaml", True),
        ("not_exist", False),
    ],
)
def test_check_path(file_name, valid, datadir):
    path = str(datadir / file_name)
    if not valid:
        with pytest.raises(FileNotFoundError):
            check_path(path)
    else:
        check_path(path)


@pytest.mark.parametrize(
    "file_name, valid",
    [
        ("autoregressive_config.yaml", True),
        ("autoregressive_config.yml", False),
    ],
)
def test_check_extension(file_name, valid, datadir):
    path = str(datadir / file_name)
    if not valid:
        with pytest.raises(ValueError):
            check_extension(path)
    else:
        check_extension(path)


@pytest.mark.parametrize(
    "file_name, path_valid, ext_valid",
    [
        ("autoregressive_config.yaml", True, True),
        ("not_exit", False, False),
        ("wrong.yml", True, False),
    ],
)
def test_check_config_path(file_name, path_valid, ext_valid, datadir):
    path = str(datadir / file_name)
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

    assert "model_type" in config
    assert "module_config" in config
    assert "data" in config
    assert "train" in config

    assert config["model_type"] == "autoregressive"
    assert config["module_config"]["img_channel"] == 3
    assert config["module_config"]["num_channels"] == 128
    assert config["module_config"]["num_layers"] == 3

    assert config["data"]["batch_size"] == 64
    assert config["data"]["num_workers"] == 4

    assert config["train"]["num_epochs"] == 10
    assert config["train"]["learning_rate"] == 0.001
    assert config["train"]["optimizer"] == "adam"
