import pytest

from gen_ai.models import GenAIModelFactory
from gen_ai.models.autoregressive import PixelCNN


@pytest.mark.parametrize(
    "model_type, config, expected",
    [
        ("autoregressive", "autoregressive_config", PixelCNN),
    ],
)
def test_gen_ai_factory(model_type, config, expected, request):
    config = request.getfixturevalue(config)
    factory = GenAIModelFactory(model_type=model_type, config=config)
    model = factory.make_model()

    assert model is not None
    assert isinstance(model, expected)
