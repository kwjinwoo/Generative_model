import pytest
import yaml

from gen_ai.configs import GenAIConfig


@pytest.fixture
def autoregressive_config(tmp_path):
    config_path = str(tmp_path / "autoregressive_config.yaml")
    data = {
        "model_type": "autoregressive",
        "module_config": {"num_channels": 128, "num_layers": 3, "img_channel": 3},
        "data": {"batch_size": 32, "num_workers": 4},
        "train": {},
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return GenAIConfig(config_path)


@pytest.fixture
def latent_variable_config(tmp_path):
    config_path = str(tmp_path / "latent_variable_config.yaml")
    data = {
        "model_type": "latent_variable",
        "module_config": {"latent_dim": 4, "img_channel": 3},
        "data": {"batch_size": 32, "num_workers": 4},
        "train": {},
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return GenAIConfig(config_path)


@pytest.fixture
def normalizaing_flow_model_config(tmp_path):
    config_path = str(tmp_path / "normalizing_flow_config.yaml")
    data = {
        "model_type": "normalizing_flow",
        "module_config": {"num_layers": 4, "hidden_dim": 512},
        "data": {"batch_size": 32, "num_workers": 4},
        "train": {},
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return GenAIConfig(config_path)


@pytest.fixture
def gan_model_config(tmp_path):
    config_path = str(tmp_path / "gan_config.yaml")
    data = {
        "model_type": "generative_adversarial_network",
        "module_config": {"noise_dim": 100},
        "data": {"batch_size": 32, "num_workers": 4},
        "train": {},
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return GenAIConfig(config_path)


@pytest.fixture
def diffusion_model_config(tmp_path):
    config_path = str(tmp_path / "diffusion_config.yaml")
    data = {
        "model_type": "diffusion",
        "module_config": {"diffusion_step": 10, "time_emb_dim": 128},
        "data": {"batch_size": 32, "num_workers": 4},
        "train": {},
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    return GenAIConfig(config_path)
