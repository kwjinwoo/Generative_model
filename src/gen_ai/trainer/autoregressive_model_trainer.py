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
        """train AutoRegressive Model.

        Args:
            model (nn.Module): torch model
            data_loader (DataLoader): train dataloader
        """
        model.train()
        model.to(self.device)

        self.optimizer = self._make_optimizer(model)
        self.criterion = nn.CrossEntropyLoss()

        print("AutoRegressive Model Training Start")
        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = x.to(self.device)
                loss = self.one_step(model, x)
                total_loss += loss.item()
            print(f"EPOCH {epoch:>3d} loss: {total_loss / len(data_loader):>6f}")
            pbar.close()
        print("AutoRegressive Model Training Finished")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """one batch training step."""
        self.optimizer.zero_grad()
        out = model(x)
        loss = self.criterion(out.permute(0, 2, 3, 1).reshape(-1, 256), (x * 255.0).reshape(-1).long())
        loss.backward()
        self.optimizer.step()
        return loss
