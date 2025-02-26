from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from gen_ai.dataset import MNISTDataset


class GenAITrainerBase(ABC):
    """Generative AI Trainer Base Class."""

    def __init__(self, dataset: MNISTDataset, config) -> None:
        """Initializes the model base class.

        Args:
            data_loader (MNISTDataset): data loader for training
            config (TrainConfig): training configuration
        """
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._optimizer = None
        self._criterion = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer for the model."""
        if self._optimizer is None:
            raise ValueError("Optimizer is not set.")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set optimizer for the model."""
        self._optimizer = optimizer

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

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer for the model.

        Args:
            model (nn.Module): model for training.

        Raises:
            ValueError: if the optimizer is invalid, raise ValueError.

        Returns:
            torch.optim.Optimizer: optimizer for the model.
        """
        if self.config.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)
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


__all__ = ["GenAITrainerBase"]
