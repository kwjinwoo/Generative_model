from __future__ import annotations

import typing

from gen_ai.configs import GenAIConfig
from gen_ai.dataset import MNISTDataset
from gen_ai.enums import ModelType
from gen_ai.models import GenAIModelBase

if typing.TYPE_CHECKING:
    from gen_ai.models.autoregressive import AutoregressiveModel
    from gen_ai.models.generative_adversarial_network import GenerativeAdversarialNetworkModel
    from gen_ai.models.latent_variable import LatentVariableModel
    from gen_ai.models.normalizing_flow import NormalizingFlowModel


def make_autoregressive_model(config: GenAIConfig, dataset: MNISTDataset) -> AutoregressiveModel:
    """Make Autoregressive Model.

    Args:
        config (GenAIConfig): Config.
        dataset (MNISTDataset): Dataset.

    Returns:
        AutoregressiveModel: Autoregressive Model.
    """
    from gen_ai.models.autoregressive import AutoregressiveModel
    from gen_ai.sampler.autoregressive_model_sampler import AutoRegressiveModelSampler
    from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer

    torch_module = AutoregressiveModel.torch_module_class(**config.module_config)
    trainer = AutoregressiveModelTrainer(config.train_config)
    sampler = AutoRegressiveModelSampler()
    return AutoregressiveModel(torch_module, trainer, sampler, dataset)


def make_latent_variable_model(config: GenAIConfig, dataset: MNISTDataset) -> LatentVariableModel:
    """Make Latent Variable Model.

    Args:
        config (GenAIConfig): Config.
        dataset (MNISTDataset): Dataset.

    Returns:
        LatentVariableModel: Latent Variable Model.
    """
    from gen_ai.models.latent_variable import LatentVariableModel
    from gen_ai.sampler.latent_variable_model_sampler import LatentVariableModelSampler
    from gen_ai.trainer.latent_variable_model_trainer import LatentVariableModelTrainer

    torch_module = LatentVariableModel.torch_module_class(**config.module_config)
    trainer = LatentVariableModelTrainer(config.train_config)
    sampler = LatentVariableModelSampler()
    return LatentVariableModel(torch_module, trainer, sampler, dataset)


def make_normalizing_flow_model(config: GenAIConfig, dataset: MNISTDataset) -> NormalizingFlowModel:
    """Make Normalizing Flow Model.

    Args:
        config (GenAIConfig): Config.
        dataset (MNISTDataset): Dataset.

    Returns:
        NormalizingFlowModel: Normalizing Flow Model.
    """
    from gen_ai.models.normalizing_flow import NormalizingFlowModel
    from gen_ai.sampler.normalizing_flow_model_sampler import NormalizingFlowModelSampler
    from gen_ai.trainer.normalizing_flow_model_trainer import NormalizingFlowModelTrainer

    torch_module = NormalizingFlowModel.torch_module_class(**config.module_config)
    trainer = NormalizingFlowModelTrainer(config.train_config)
    sampler = NormalizingFlowModelSampler()
    return NormalizingFlowModel(torch_module, trainer, sampler, dataset)


def make_generative_adversarial_network_model(
    config: GenAIConfig, dataset: MNISTDataset
) -> GenerativeAdversarialNetworkModel:
    """Make Generative Adversarial Network Model.

    Args:
        config (GenAIConfig): Config.
        dataset (MNISTDataset): Dataset.

    Returns:
        None: Generative Adversarial Network Model.
    """
    from gen_ai.models.generative_adversarial_network import GenerativeAdversarialNetworkModel
    from gen_ai.sampler.generative_adversarial_network_sampler import GenerativeAdversarialNetworkSampler
    from gen_ai.trainer.generative_adversarial_network_trainer import GenerativeAdversarialNetworkTrainer

    torch_module = GenerativeAdversarialNetworkModel.torch_module_class(**config.module_config)
    trainer = GenerativeAdversarialNetworkTrainer(config.train_config)
    sampler = GenerativeAdversarialNetworkSampler()
    return GenerativeAdversarialNetworkModel(torch_module, trainer, sampler, dataset)


class GenAIModelFactory:
    """Generative AI model factory."""

    def __init__(self, config: GenAIConfig) -> None:
        """Initializes the model factory.

        Args:
            config (GenAIConfig): configuration
        """
        self.config = config
        self.model_maker_map = {
            ModelType.autoregressive: make_autoregressive_model,
            ModelType.latent_variable: make_latent_variable_model,
            ModelType.normalizing_flow: make_normalizing_flow_model,
            ModelType.generative_adversarial_network: make_generative_adversarial_network_model,
        }

    def make_model(self) -> GenAIModelBase:
        """creates a model based on the model type.

        Raises:
            ValueError: Unsupported model type

        Returns:
            GenAIModelBase: model
        """
        dataset = MNISTDataset(self.config.data_config)
        if self.config.model_type in self.model_maker_map:
            maker = self.model_maker_map[self.config.model_type]
            return maker(self.config, dataset)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
