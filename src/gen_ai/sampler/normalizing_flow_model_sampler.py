import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Distribution
from torch.utils.data import Dataset


def get_random_sample(dataset: Dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """get random sample for latent space interpolation.
    it sample two data from dataset randomly. if two samples are different class, return data.

    Args:
        dataset (Dataset): dataset
        device (torch.device): device

    Raises:
        RuntimeError: if the number of iter is exceed max iter.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: randome two data.
    """
    MAX_ITER = 10
    iter = 0
    while MAX_ITER >= iter:
        random_idx = torch.randint(0, len(dataset), (2,))
        x1, class1 = dataset[random_idx[0]]
        x2, class2 = dataset[random_idx[1]]
        if class1 != class2:
            return x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device)
        iter += 1
    raise RuntimeError("Exceed MAX ITER.")


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
        self.interpolate_latent_space(model, dataset, save_dir)

    def random_sample(self, model: nn.Module, save_dir: str, num_samples: int) -> None:
        """sample image from random latent variable."""
        print("Normalizing Flow Model Random Sampling Start.")
        with torch.no_grad():
            z = self.prior.sample(torch.Size([num_samples]))

            generated, _ = model(z, reverse=True)
            generated = generated.clip(0, 1).view(-1, 1, 28, 28)

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

    def interpolate_latent_space(self, model: nn.Module, dataset: Dataset, save_dir: str) -> None:
        """Interpolate Latent Space."""
        print("Normalizing Flow Model Interpolating Start.")
        x1, x2 = get_random_sample(dataset, self.device)
        with torch.no_grad():
            z1, _ = model(x1, reverse=False)
            z2, _ = model(x2, reverse=False)

            alphas = torch.linspace(0, 1, 10).to(self.device)
            interpolated_z = torch.stack([(1 - alpha) * z1 + alpha * z2 for alpha in alphas]).squeeze(1)

            generated, _ = model(interpolated_z, reverse=True)
            generated = generated.clip(0, 1).view(-1, 1, 28, 28)

        fig, axes = plt.subplots(1, 10, figsize=(10, 2))
        for i in range(10):
            axes[i].imshow(generated[i].squeeze().cpu(), cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Normalizing Flow Model Interpolating")
        plt.savefig(os.path.join(save_dir, "NICE_interpolate.png"))
        print(f"Normalizing Flow Model Interpolating Finished. saved at {save_dir}")
