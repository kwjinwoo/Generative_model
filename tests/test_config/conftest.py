import pytest
import yaml


@pytest.fixture
def autoregressive_config_file(datadir) -> dict:
    with open(datadir / "autoregressive_config.yaml") as f:
        return yaml.safe_load(f)
