import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.utils.data import DataLoader, Dataset


class NormalizingFlowModelSampler:
    """NormalizingFlowModelSampler class to from normalizing flow models."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prior = None

    @property
    def prior(self) -> Distribution:
        """Prior of Normalizing Flow Model."""
        if self._prior is None:
            raise ValueError("Prior is not set.")
        return self._prior

    @prior.setter
    def prior(self, prior: Distribution):
        """Set Prior for Normalizing Flow Model."""
        self._prior = prior

    def sample(self, model: nn.Module, dataset: Dataset, save_dir: str, num_samples: int) -> None:
        """sample method to sample from normalizing flow models.

        Args:
            model (nn.Module): normalizing flow model.
            dataset (Dataset): valid dataset.
            save_dir (str): saved directory.
            num_samples (int): the number of samples.
        """
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        model.eval()
        model.to(self.device)

        self.random_sample(model, save_dir, num_samples)
        self.reconstruct(model, dataset, save_dir)

    def random_sample(self, model: nn.Module, save_dir: str, num_samples: int) -> None:
        """sample image from random latent variable."""
        print("Normalizing Flow Model Random Sampling Start.")
        with torch.no_grad():
            z = self.prior.sample(torch.Size([num_samples, 28 * 28]))

            generated, _ = model(z, reverse=True)
            generated = generated.view(-1, 1, 28, 28)

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")

        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("NICE generated samples")
        plt.savefig(os.path.join(save_dir, "NICE_generated.png"))
        print(f"Normalizing Flow Model Finished. saved at {save_dir}")

    def reconstruct(self, model: nn.Module, dataset: Dataset, save_dir: str) -> None:
        valid_loader = DataLoader(dataset, batch_size=8)
        print("Normalizing Flow Model Reconstructing Strat.")

        origin = next(iter(valid_loader))[0].to(self.device)
        with torch.no_grad():
            z, _ = model(origin, reverse=False)
            recon, _ = model(z, reverse=True)
        recon = recon.view(-1, 1, 28, 28)
        fig, axes = plt.subplots(2, 8, figsize=(8, 2))
        for i in range(8):
            axes[0, i].imshow(origin[i].cpu().squeeze(), cmap="gray")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
            axes[1, i].axis("off")
        plt.suptitle("Original (Top) vs. Reconstructed (Bottom)", fontsize=16)
        plt.savefig(os.path.join(save_dir, "NICE_reconstruct.png"))
        print(f"Normalizing Flow Model Reconstructing Finished. saved at {save_dir}")
