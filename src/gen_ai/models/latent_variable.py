from __future__ import annotations

import os
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_ai.models import GenAIModelBase

if typing.TYPE_CHECKING:
    from gen_ai.dataset import MNISTDataset
    from gen_ai.sampler.latent_variable_model_sampler import LatentVariableModelSampler
    from gen_ai.trainer.latent_variable_model_trainer import LatentVariableModelTrainer


class Encoder(nn.Module):
    """Encoder for Convolutional Variational AutoEncoder."""

    def __init__(self, input_channel: int, latent_dim: int) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.mu_linear = nn.Linear(in_features=128 * 4 * 4, out_features=latent_dim)
        self.log_var_linear = nn.Linear(in_features=128 * 4 * 4, out_features=latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """encode input tensor to latent variable.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean and log variance of latent variable.
        """
        x = self.layers(inputs)
        x = self.flatten(x)
        mean = self.mu_linear(x)
        log_var = self.log_var_linear(x)
        return mean, log_var


class Decoder(nn.Module):
    """Decoder for Convolutional Variational AutoEncoder."""

    def __init__(self, latent_dim: int, output_channel: int) -> None:
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.linear = nn.Linear(in_features=latent_dim, out_features=128 * 4 * 4)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=output_channel,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """decode latent variable to output tensor.

        Args:
            inputs (torch.Tensor): latent variable.

        Returns:
            torch.Tensor: decoded output tensor.
        """
        x = self.linear(inputs)
        x = x.view(-1, 128, 4, 4)
        return self.layers(x)


class ConvolutionalVAE(nn.Module):
    """Convolutional Variational AutoEncoder."""

    def __init__(self, img_channel: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(img_channel, latent_dim)
        self.decoder = Decoder(latent_dim, 256)

    def reparameterizing(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """reparameterizing trick for VAE.
        it sample latent variable from normal distribution. it is calculated as mean + exp(log_var / 2) * eps.

        Args:
            mean (torch.Tensor): mean of latent variable.
            log_var (torch.Tensor): log variance of latent variable.

        Returns:
            torch.Tensor: reparameterized latent variable.
        """
        eps = torch.randn(mean.size(), device=mean.device)
        return eps * torch.exp(log_var * 0.5) + mean

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward method for Convolutional Variational AutoEncoder.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: output tensor, mean of latent variable, log variance of
                                                            latent variable.
        """
        mean, log_var = self.encoder(inputs)
        z = self.reparameterizing(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var


def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """ELBO loss function.
    it is sum of reconst_loss and KL divergence. reconst_loss is binary cross entropy loss between x and x_hat.
    KL divergence is Kullback-Leibler divergence between N(mean, log_var) and N(0, 1). it is calculated as
    -0.5 * sum(1 + log_var - mean^2 - exp(log_var)). it is used to make latent variable to be close to N(0, 1).

    Args:
        x (torch.Tensor): True image tensor.
        x_hat (torch.Tensor): Decoder's output tensor.
        mean (torch.Tensor): Latent variable's mean tensor.
        log_var (torch.Tensor): Latent variable's log variance tensor.

    Returns:
        torch.Tensor: ELBO loss.
    """
    reconst_loss = F.cross_entropy(x_hat.permute(0, 2, 3, 1).reshape(-1, 256), x.reshape(-1).long(), reduction="sum")
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconst_loss + kl_div


class LatentVariableModel(GenAIModelBase):
    torch_module_class = ConvolutionalVAE

    def __init__(
        self,
        torch_module: ConvolutionalVAE,
        trainer: LatentVariableModelTrainer,
        sampler: LatentVariableModelSampler,
        dataset: MNISTDataset,
    ) -> None:
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        self.trainer.criterion = elbo_loss
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir: str, num_samples: int):
        self.sampler.sample(self.torch_module, self.dataset.valid_dataset, save_dir, num_samples)

    def load(self, file_path) -> None:
        """load trained model from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        self.torch_module.load_state_dict(torch.load(file_path, map_location=self.sampler.device))

    def save(self, save_dir: str) -> None:
        """save trained model to save dir.

        Args:
            save_dir (str): svae directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.torch_module.state_dict(), os.path.join(save_dir, f"{self.torch_module_class.__name__}.pth"))
