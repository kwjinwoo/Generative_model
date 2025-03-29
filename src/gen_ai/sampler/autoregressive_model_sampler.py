import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class AutoRegressiveModelSampler:
    """AtuoRegressiveModelSampler class to sample from autoregressive models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model: nn.Module, dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        """sample method to sample from autoregressive models.

        Args:
            model (nn.Module): autoregressive model
            dataset (Dataset): valid dataset
            saved_dir (str): svaed directory
            num_samples (int): the number of samples
        """
        if os.path.exists(saved_dir) is False:
            os.makedirs(saved_dir)
        model.eval()
        model.to(self.device)

        self.full_sample(model, saved_dir, num_samples)
        self.half_sample(model, dataset, saved_dir, num_samples)

    def full_sample(self, model: nn.Module, saved_dir: str, num_samples: int) -> None:
        """sample full image from autoregressive model."""
        generated = torch.zeros((num_samples, 1, 28, 28), dtype=torch.float32)
        generated = generated.to(self.device)
        print("AutoRegressive Full Sampling Start.")
        with torch.no_grad():
            for h in range(28):
                for w in range(28):
                    out = model(generated)
                    out = torch.softmax(out[:, :, h, w], dim=1)
                    generated_pixel = torch.multinomial(out, num_samples=1)
                    generated[:, :, h, w] = generated_pixel

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")
        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("PixelCNN generated samples")
        plt.savefig(os.path.join(saved_dir, "pixelCNN_generate.png"))
        print(f"AutoRegressive Full Sampling Finished. saved at {saved_dir}")

    def half_sample(self, model: nn.Module, valid_dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        """sample half of the image from autoregressive model."""
        valid_loader = DataLoader(valid_dataset, batch_size=num_samples)
        generated = next(iter(valid_loader))[0].to(self.device)
        generated = generated * 255.0
        generated[:, :, 14:, :] = 0
        mask = np.zeros(list(generated.shape[2:]))
        mask[14:, :] = 1
        print("AutoRegressive Half Sampling Start.")
        with torch.no_grad():
            for h in range(14, 28):
                for w in range(28):
                    out = model(generated)
                    out = torch.softmax(out[:, :, h, w], dim=1)
                    generated_pixel = torch.multinomial(out, num_samples=1)
                    generated[:, :, h, w] = generated_pixel
        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")
        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.imshow(mask, cmap="Reds", alpha=0.3)
            plt.axis("off")
        plt.suptitle("PixelCNN half generated samples")
        plt.savefig(os.path.join(saved_dir, "pixelCNN_half_generate.png"))
        print(f"AutoRegressive Half Sampling Finished. saved at {saved_dir}")
