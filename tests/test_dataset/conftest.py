import pytest

from gen_ai.configs import GenAIConfig


@pytest.fixture
def test_data_config(shared_datadir):
    path = shared_datadir / "test.yaml"
    return GenAIConfig(str(path)).data_config
