import pytest

from gen_ai.factory import GenAIModelFactory
from gen_ai.models.autoregressive import AutoregressiveModel


@pytest.mark.skip(reason="Downloading MNIST dataset is slow.")
@pytest.mark.parametrize(
    "model_type, config, expected",
    [
        ("autoregressive", "autoregressive_config", AutoregressiveModel),
    ],
)
def test_model_factory(model_type, config, expected, request):
    config = request.getfixturevalue(config)
    factory = GenAIModelFactory(model_type=model_type, config=config)
    model = factory.make_model()

    assert model is not None
    assert isinstance(model, expected)
