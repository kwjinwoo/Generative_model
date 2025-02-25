import torch
import torch.nn as nn
from tqdm import tqdm

from gen_ai.configs.train_configs import TrainConfig
from gen_ai.dataset import MNISTDataset
from gen_ai.trainer import GenAITrainerBase


class AutoregressiveModelTrainer(GenAITrainerBase):
    """Trainer for AutoRegressive Model."""

    def __init__(self, dataset: MNISTDataset, config: TrainConfig) -> None:
        super().__init__(dataset, config)

    def train(self, model: nn.Module) -> None:
        model.train()
        model.to(self.device)

        self.optimizer = self._get_optimizer(model)
        self.criterion = nn.BCELoss()
        data_loader = self.dataset.loader
        
        print("Training Start")
        for epoch in range(self.config.num_epochs):
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
