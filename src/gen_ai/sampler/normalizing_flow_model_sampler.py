import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NormalizingFlowModelSampler:
    """NormalizingFlowModelSampler class to from normalizing flow models."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def random_sample(self, model: nn.Module, save_dir: str, num_samples: int) -> None:
        """sample image from random latent variable."""
        print("Normalizing Flow Model Random Sampling Start.")
        with torch.no_grad():
            z = torch.randn(num_samples, 784, device=self.device)

            generated, _ = model(z, reverse=False)
            generated = torch.sigmoid(generated.view(-1, 1, 28, 28))

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")

        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("RealNVP generated samples")
        plt.savefig(os.path.join(save_dir, "RealNVP_generated.png"))
        print(f"Normalizing Flow Model Finished. saved at {save_dir}")
