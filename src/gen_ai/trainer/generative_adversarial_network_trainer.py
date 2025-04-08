import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_ai.trainer import GenAITrainerBase


class GenerativeAdversarialNetworkTrainer(GenAITrainerBase):
    """Generative Adversarial Network Trainer."""

    def __init__(self, config: dict[str, int | float | str]) -> None:
        """Initializes the GAN trainer.

        Args:
            config (dict[str, int | float | str]): training configuration
        """
        super().__init__(config)

    def train(self, model: nn.Module, data_loader: DataLoader) -> None:
        """Train the GAN model.

        Args:
            model (nn.Module): GAN model
            data_loader (DataLoader): train dataloader
        """
        model.train()
        model.to(self.device)

        generator_optimizer = self._make_optimizer(model.generator)
        discriminator_optimizer = self._make_optimizer(model.discriminator)

        self.set_optimizer("generator", generator_optimizer)
        self.set_optimizer("discriminator", discriminator_optimizer)
        self.criterion = nn.BCELoss()
        print("GAN Training Start")
        for epoch in range(self.config["num_epochs"]):
            total_generator_loss = 0
            total_discriminator_loss = 0
            pbar = tqdm(data_loader, total=len(data_loader))
            for x, _ in pbar:
                x = x.to(self.device)
                generator_loss, discriminator_loss = self.one_step(model, x)
                total_generator_loss += generator_loss.item()
                total_discriminator_loss += discriminator_loss.item()
            print(
                f"EPOCH {epoch:>3d} generator loss: {total_generator_loss / len(data_loader):>6f} discriminator loss: "
                f"{total_discriminator_loss / len(data_loader):>6f}"
            )
            pbar.close()
        print("GAN Training Finished")

    def one_step(self, model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One batch training step."""
        return self._train_generator(model, x), self._train_discriminator(model, x)

    def _train_generator(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Train generator.

        Args:
            model (nn.Module): GAN model
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: generator loss
        """
        fake_output = model(x, generator_only=True)
        optimizer = self.get_optimizer("generator")
        optimizer.zero_grad()
        generator_loss = self.criterion(fake_output, torch.ones_like(fake_output))
        generator_loss.backward(retain_graph=True)
        optimizer.step()
        return generator_loss

    def _train_discriminator(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Train discriminator.

        Args:
            model (nn.Module): GAN model
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: discriminator loss
        """
        fake_output, real_output = model(x, generator_only=False)
        optimizer = self.get_optimizer("discriminator")
        optimizer.zero_grad()
        real_loss = self.criterion(real_output, torch.ones_like(real_output))
        fake_loss = self.criterion(fake_output, torch.zeros_like(fake_output))
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        optimizer.step()
        return discriminator_loss
