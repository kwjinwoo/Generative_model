import pytest

from gen_ai.factory import GenAIModelFactory
from gen_ai.models.autoregressive import AutoregressiveModel
from gen_ai.models.diffusion import DiffusionModel
from gen_ai.models.generative_adversarial_network import GenerativeAdversarialNetworkModel
from gen_ai.models.latent_variable import LatentVariableModel
from gen_ai.models.normalizing_flow import NormalizingFlowModel


@pytest.mark.parametrize(
    "config, expected",
    [
        ("autoregressive_config", AutoregressiveModel),
        ("latent_variable_config", LatentVariableModel),
        ("normalizaing_flow_model_config", NormalizingFlowModel),
        ("gan_model_config", GenerativeAdversarialNetworkModel),
        ("diffusion_model_config", DiffusionModel),
    ],
)
def test_model_factory(config, expected, request):
    config = request.getfixturevalue(config)
    factory = GenAIModelFactory(config=config)
    model = factory.make_model()

    assert model is not None
    assert isinstance(model, expected)
