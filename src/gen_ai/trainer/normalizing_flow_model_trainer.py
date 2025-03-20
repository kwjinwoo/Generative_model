from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


def dequantize(x, alpha=1e-5):
    return (x + torch.rand_like(x) * alpha).clamp(0, 1)


class NormalizingFlowModelTrainer(GenAITrainerBase):
    """Trainer for Normalizing Flow Model."""

    def __init__(self, config: dict[str, int | float | str]) -> None:
        super().__init__(config)
        self._criterion = None

    @property
    def criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Creterion for Normalizing Flow Model."""
        if self._criterion is None:
            raise ValueError("Criterion is not set.")
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: Callable) -> None:
        """Set criterion for Normalizing Flow Model."""
        self._criterion = criterion

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        """train Normalizing Flow Model.

        Args:
            model (nn.Module): torch model.
            data_loader (DataLoader): train dataloader.
        """
        model.train()
        model.to(self.device)

        self.optimizer = self._get_optimizer(model)

        print("Normalizing Flow Model Training Start.")
        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = dequantize(x).to(self.device)

                loss = self.one_step(model, x)
                total_loss += loss.item()
            print(f"EPOCH {epoch:>3d} loss: {total_loss / len(data_loader):>6f}")
            pbar.close()
        print("Normalizing Flow Model Traning Finished.")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """one batch training step."""
        self.optimizer.zero_grad()
        z, log_det_jacobian = model(x)
        loss = self.criterion(z, log_det_jacobian)
        loss.backward()
        self.optimizer.step()
        return loss
