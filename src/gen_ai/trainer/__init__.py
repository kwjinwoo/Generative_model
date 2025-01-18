from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from gen_ai.configs import TrainConfig
from gen_ai.dataset import MNISTLoader

from .autoregressive_model_trainer import AutoregressiveModelTrainer


class GenAITrainerBase(ABC):
    """Generative AI Trainer Base Class."""

    def __init__(self, model: nn.Module, data_loader: MNISTLoader, config: TrainConfig) -> None:
        """Initializes the model base class.

        Args:
            model (nn.Module): model to train
            data_loader (MNISTLoader): data loader for training
            config (TrainConfig): training configuration
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config

        self._optimizer = self._get_optimizer()
        self._criterion = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer for the model."""
        return self._optimizer

    @property
    def criterion(self) -> nn.Module:
        """Get criterion for the model."""
        if self._criterion is None:
            raise ValueError("Criterion is not set.")
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: nn.Module) -> None:
        """Set criterion for the model."""
        self._criterion = criterion

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer for the model.

        Raises:
            ValueError: if the optimizer is invalid, raise ValueError.

        Returns:
            torch.optim.Optimizer: optimizer for the model.
        """
        if self.config.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer {self.config.optimizer}")

    @abstractmethod
    def train(self, device: torch.device) -> None:
        """Train the model."""
        pass

    def save(self, save_path: str) -> None:
        """Save the model.

        Args:
            save_dir (str): directory to save the model.
        """
        torch.save(self.model.state_dict(), save_path)


__all__ = ["GenAITrainerBase", "AutoregressiveModelTrainer"]
