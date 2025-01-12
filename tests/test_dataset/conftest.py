import pytest

from gen_ai.configs import GenAIConfig


@pytest.fixture
def test_data_config():
    return GenAIConfig(
        data_config={"batch_size": 32, "num_workers": 4},
        model_config={},
        train_config={},
    )
