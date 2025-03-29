from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


class LatentVariableModelTrainer(GenAITrainerBase):
    """Trainer for Latent Variable Model."""

    def __init__(self, config: dict[str, int | float | str]) -> None:
        super().__init__(config)
        self._criterion = None

    @property
    def criterion(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Criterion for Latent Variable Model."""
        if self._criterion is None:
            raise ValueError("Criterion is not set.")
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: Callable) -> None:
        """Set criterion for Latent Variable Model."""
        self._criterion = criterion

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        """train Latent Variable Model.

        Args:
            model (nn.Module): torch model
            data_loader (DataLoader): train dataloader
        """
        model.train()
        model.to(self.device)

        self.optimizer = self._get_optimizer(model)

        print("Latent Variable Model Training Start")
        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = x.to(self.device)

                loss = self.one_step(model, x)
                total_loss += loss.item()
            print(f"EPOCH {epoch:>3d} ELBO: {- total_loss / len(data_loader):>6f}")
            pbar.close()

        print("Latent Variable Model Training Finished")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """one batch training step."""
        self.optimizer.zero_grad()
        out, mean, log_var = model(x)
        loss = self.criterion(x * 255.0, out, mean, log_var)
        loss.backward()
        self.optimizer.step()
        return loss
