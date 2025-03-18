import pytest

from gen_ai.factory import GenAIModelFactory
from gen_ai.models.autoregressive import AutoregressiveModel
from gen_ai.models.latent_variable import LatentVariableModel
from gen_ai.models.normalizing_flow import NormalizingFlowModel


@pytest.mark.parametrize(
    "config, expected",
    [
        ("autoregressive_config", AutoregressiveModel),
        ("latent_variable_config", LatentVariableModel),
        ("normalizaing_flow_model_config", NormalizingFlowModel),
    ],
)
def test_model_factory(config, expected, request):
    config = request.getfixturevalue(config)
    factory = GenAIModelFactory(config=config)
    model = factory.make_model()

    assert model is not None
    assert isinstance(model, expected)
