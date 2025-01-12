from abc import ABC, abstractmethod

from gen_ai.configs import GenAIConfig
from gen_ai.enums import ModelType
from gen_ai.models.autoregressive import AutoregressiveModel, PixelCNN


class GenAIModelBase(ABC):
    """Generative AI model base class."""

    def __init__(self, trainer, sampler) -> None:
        """Initializes the model base class.

        Args:
            trainer (_type_): _description_
            sampler (_type_): _description_
        """
        self.trainer = trainer
        self.sampler = sampler

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def sample(self) -> None:
        """Sample from the model."""
        pass


class GenAIModelFactory:
    """Generative AI model factory."""

    def __init__(self, model_type: ModelType, config: GenAIConfig) -> None:
        """Initializes the model factory.

        Args:
            model_type (ModelType): model type
            config (GenAIConfig): configuration
        """
        self.model_type = model_type
        self.config = config

    def make_model(self) -> GenAIModelBase:
        """creates a model based on the model type.

        Raises:
            ValueError: Unsupported model type

        Returns:
            nn.Module: model
        """
        if self.model_type == ModelType.autoregressive:
            model = PixelCNN(**self.config.model_config)
            trainer = None
            sampler = None
            return AutoregressiveModel(model, trainer, sampler)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


__all__ = ["GenAIModelFactory"]
