import torch.nn as nn

from gen_ai.configs import GenAIConfig
from gen_ai.enums import ModelType
from gen_ai.models.autoregressive import PixelCNN


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

    def make_model(self) -> nn.Module:
        """creates a model based on the model type.

        Raises:
            ValueError: Unsupported model type

        Returns:
            nn.Module: model
        """
        if self.model_type == ModelType.autoregressive:
            return PixelCNN(**self.config.model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


__all__ = ["GenAIModelFactory"]
