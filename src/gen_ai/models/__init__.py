from abc import ABC, abstractmethod

from gen_ai.configs import GenAIConfig
from gen_ai.dataset import MNISTLoader
from gen_ai.enums import ModelType
from gen_ai.models.autoregressive import AutoregressiveModel, PixelCNN
from gen_ai.trainer import AutoregressiveModelTrainer, GenAITrainerBase


class GenAIModelBase(ABC):
    """Generative AI model base class."""

    def __init__(self, trainer: GenAITrainerBase, sampler) -> None:
        """Initializes the model base class.

        Args:
            trainer (GenAITrainerBase): model trainer
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

    @abstractmethod
    def save(self, save_dir: str) -> None:
        """Save the model.

        Args:
            save_dir (str): save_dir
        """
        save_path = f"{save_dir}/{self.__class__.__name__}.pt"
        self.trainer.save(save_path)
        pass

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load the model.

        Args:
            file_path (str): file path
        """
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
            torch_module = PixelCNN(**self.config.model_config)
            trainer = AutoregressiveModelTrainer(MNISTLoader(self.config.data_config), self.config.train_config)
            sampler = None
            return AutoregressiveModel(torch_module, trainer, sampler)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


__all__ = ["GenAIModelFactory", "GenAIModelBase"]
