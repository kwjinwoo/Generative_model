import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


class AutoregressiveModelTrainer(GenAITrainerBase):
    """Trainer for AutoRegressive Model."""

    def __init__(self, config: dict[str, int | float | str]) -> None:
        super().__init__(config)

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        model.train()
        model.to(self.device)

        self.optimizer = self._get_optimizer(model)
        self.criterion = nn.BCELoss()

        print("Training Start")
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.config["num_epochs"]):
                mean_loss = 0
                pbar = tqdm(data_loader)
                for x, _ in pbar:
                    x = x.to(self.device)

                    loss = self.one_step(model, x)
                    mean_loss += loss.item()
                pbar.desc = f"EPOCH {epoch:>3d} loss: {mean_loss / len(data_loader):>6f}"
        pbar.close()
        print("Training Finished")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        out = model(x)
        loss = self.criterion(out, x)
        loss.backward()
        self.optimizer.step()
        return loss
