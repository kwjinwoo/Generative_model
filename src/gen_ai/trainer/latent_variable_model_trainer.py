from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


class LatentVariableModelTrainer(GenAITrainerBase):
    def __init__(self, config: dict[str, int | float | str]) -> None:
        super().__init__(config)
        self._criterion = None

    @property
    def criterion(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if self._criterion is None:
            raise ValueError("Criterion is not set.")
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: Callable) -> None:
        self._criterion = criterion

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        model.train()
        model.to(self.device)

        self.optimizer = self._get_optimizer(model)

        print("Latent Variable Model Training Start")
        for epoch in range(self.config["num_epochs"]):
            mean_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = x.to(self.device)

                loss = self.one_step(model, x)
                mean_loss += loss.item()
            print(f"EPOCH {epoch:>3d} ELBO: {- mean_loss / len(data_loader):>6f}")
            pbar.close()

        print("Latent Variable Model Training Finished")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        out, mean, log_var = model(x)
        loss = self.criterion(x, out, mean, log_var)
        loss.backward()
        self.optimizer.step()
        return loss
