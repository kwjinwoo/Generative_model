import pytest

from gen_ai.configs import GenAIConfig


@pytest.fixture
def autoregressive_config():
    return GenAIConfig(
        model_config={"num_channels": 128, "num_layers": 3, "img_channel": 3},
        data_config={},
        train_config={},
    )
