import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LatentVariableModelSampler:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model: nn.Module, dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        if os.path.exists(saved_dir) is False:
            os.makedirs(saved_dir)
        model.eval()
        model.to(self.device)

        print("Latent Variable Sampling Start.")
        latent_dim = model.latent_dim
        with torch.no_grad():
            eps = torch.randn((num_samples, latent_dim), device=self.device)

            generated = model.decoder(eps)
            generated = torch.bernoulli(generated)

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")

        print(generated.shape)
        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("CVAE generated samples")
        plt.savefig(os.path.join(saved_dir, "CVAE_generate.png"))
        print(f"Latent Variable Sampling Finished. saved at {saved_dir}")
