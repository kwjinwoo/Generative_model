from abc import ABC, abstractmethod

import torch.nn as nn

from gen_ai.trainer import GenAITrainerBase


class GenAIModelBase(ABC):
    """Generative AI model base class."""

    def __init__(self, torch_module: nn.Module, trainer: GenAITrainerBase, sampler) -> None:
        """Initializes the model base class.

        Args:
            torch_module (nn.Module): torch module
            trainer (GenAITrainerBase): model trainer
            sampler (_type_): _description_
        """
        self.torch_module = torch_module
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


__all__ = ["GenAIModelBase"]
