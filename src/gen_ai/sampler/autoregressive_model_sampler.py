import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class AutoRegressiveModelSampler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model: nn.Module, dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        if os.path.exists(saved_dir) is False:
            os.makedirs(saved_dir)
        model.eval()
        model.to(self.device)

        self.full_sample(model, saved_dir, num_samples)
        self.half_sample(model, dataset, saved_dir, num_samples)

    def full_sample(self, model: nn.Module, saved_dir: str, num_samples: int) -> None:
        generated = torch.zeros((num_samples, 1, 28, 28), dtype=torch.float32)
        generated = generated.to(self.device)
        print("Sampling Start.")
        with torch.no_grad():
            for h in range(28):
                for w in range(28):
                    out = model(generated)
                    generated_pixel = torch.bernoulli(out[:, :, h, w])
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

    def half_sample(self, model: nn.Module, valid_dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        valid_loader = DataLoader(valid_dataset, batch_size=num_samples)
        generated = next(iter(valid_loader))[0].to(self.device)
        generated[:, :, 14:, :] = 0
        with torch.no_grad():
            for h in range(14, 28):
                for w in range(28):
                    out = model(generated)
                    generated_pixel = torch.bernoulli(out[:, :, h, w])
                    generated[:, :, h, w] = generated_pixel * 2
        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")
        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("PixelCNN half generated samples")
        plt.savefig(os.path.join(saved_dir, "pixelCNN_half_generate.png"))
