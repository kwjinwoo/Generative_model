import torch
import torch.nn as nn
from tqdm import tqdm

from gen_ai.configs.train_configs import TrainConfig
from gen_ai.dataset import MNISTLoader
from gen_ai.trainer import GenAITrainerBase


class AutoregressiveModelTrainer(GenAITrainerBase):
    """Trainer for AutoRegressive Model."""

    def __init__(self, model: nn.Module, data_loader: MNISTLoader, config: TrainConfig) -> None:
        super().__init__(model, data_loader, config)

        self._criterion = nn.BCELoss()

    def train(self, device: torch.device) -> None:
        self.model.train()
        self.model.to(device)

        print("Training Start")
        for epoch in range(self.config.num_epochs):
            mean_loss = 0
            pbar = tqdm(self.data_loader)
            for x, _ in pbar:
                x = x.to(device)

                loss = self.one_step(x)
                mean_loss += loss.item()
            pbar.desc = f"EPOCH {epoch:>3d} loss: {mean_loss / len(self.data_loader):>6f}"
        pbar.close()
        print("Training Finished")

    def one_step(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        self.optimizer.step()
        return loss
