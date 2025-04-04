from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class OptimizerManager:
    """Manages one or multiple optimizers for training models."""

    def __init__(self):
        self._optimizers = {}

    def set_optimizer(self, name: str, optimizer: Optimizer) -> None:
        """Set optimizer for the model."""
        self._optimizers[name] = optimizer

    def get_optimizer(self, name: str) -> Optimizer:
        """Get optimizer for the model."""
        if name not in self._optimizers:
            raise ValueError(f"Optimizer {name} is not set.")
        return self._optimizers[name]

    @property
    def optimizer(self) -> Optimizer:
        """Get the default optimizer."""
        if "default" not in self._optimizers:
            raise ValueError("Default optimizer is not set.")
        return self._optimizers["default"]

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the default optimizer."""
        self._optimizers["default"] = optimizer


class GenAITrainerBase(ABC):
    """Generative AI Trainer Base Class."""

    def __init__(self, config: dict[str, int | float | str]) -> None:
        """Initializes the model base class.

        Args:
            config (dict[str, int | float | str]): training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_manager = OptimizerManager()

        self._criterion = None

    @property
    def optimizer(self) -> Optimizer:
        """Get optimizer for the model."""
        return self.optimizer_manager.optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set optimizer for the model."""
        self.optimizer_manager.optimizer = optimizer

    def get_optimizer(self, name: str) -> Optimizer:
        """Get optimizer by name."""
        return self.optimizer_manager.get_optimizer(name)

    def set_optimizer(self, name: str, optimizer: Optimizer) -> None:
        """Set optimizer by name."""
        self.optimizer_manager.set_optimizer(name, optimizer)

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

    def _make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """make optimizer for the model.

        Args:
            model (nn.Module): model for training.

        Raises:
            ValueError: if the optimizer is invalid, raise ValueError.

        Returns:
            torch.optim.Optimizer: optimizer for the model.
        """
        optimizer_name = self.config["optimizer"]
        if optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        elif optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config["learning_rate"])
        else:
            raise ValueError(f"Invalid optimizer {optimizer_name}")

    @abstractmethod
    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        """Train the model."""
        pass


__all__ = ["GenAITrainerBase"]
