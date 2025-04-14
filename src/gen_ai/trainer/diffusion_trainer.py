import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


class DiffusionTrainer(GenAITrainerBase):
    def __init__(self, config: dict[str, int | float | str]) -> None:
        """Initializes the Diffusion trainer.

        Args:
            config (dict[str, int | float | str]): training configuration
        """
        super().__init__(config)

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        """Train the diffusion model.

        Args:
            model (nn.Module): diffusion model
            data_loader (DataLoader): train dataloader
        """
        model.train()
        model.to(self.device)

        self.optimizer = self._make_optimizer(model)
        self.criterion = nn.MSELoss()

        print("Diffusion Model Training Start")
        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = x.to(self.device)
                loss = self.one_step(model, x)
                total_loss += loss.item()
            print(f"EPOCH {epoch:>3d} Loss: {total_loss / len(data_loader):>6f}")
            pbar.close()

    def one_step(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """One batch training step."""
        self.optimizer.zero_grad()
        t = torch.randint(0, model.diffusion_step, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x)
        x_t = model.q_sample(x, t, noise)
        noise_pred = model(x_t, t)
        loss = self.criterion(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        return loss
