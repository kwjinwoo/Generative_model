import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LatentVariableModelSampler:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model: nn.Module, dataset: Dataset, saved_dir: str, num_samples: int) -> None:
        if os.path.exists(saved_dir) is False:
            os.makedirs(saved_dir)
        model.eval()
        model.to(self.device)

        self.random_sample(model, saved_dir, num_samples)
        self.reconstruct(model, dataset, saved_dir)

    def random_sample(self, model: nn.Module, saved_dir: str, num_samples: int) -> None:
        print("Latent Variable Random Sampling Start.")
        latent_dim = model.latent_dim
        with torch.no_grad():
            eps = torch.randn((num_samples, latent_dim), device=self.device)

            generated = model.decoder(eps)
            generated = torch.bernoulli(generated)

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")

        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("VAE generated samples")
        plt.savefig(os.path.join(saved_dir, "VAE_generate.png"))
        print(f"Latent Variable Sampling Finished. saved at {saved_dir}")

    def reconstruct(self, model: nn.Module, dataset: Dataset, save_dir: str) -> None:
        valid_loader = DataLoader(dataset, batch_size=8)
        print("Latent Variable Reconstructing Start")

        origin = next(iter(valid_loader))[0].to(self.device)
        with torch.no_grad():
            mean, _ = model.encoder(origin)
            recon = model.decoder(mean)
            recon = torch.bernoulli(recon)

        fig, axes = plt.subplots(2, 8, figsize=(8, 2))
        for i in range(8):
            axes[0, i].imshow(origin[i].cpu().squeeze(), cmap="gray")
            axes[0, i].axis("off")

            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap="gray")
            axes[1, i].axis("off")
        plt.suptitle("Original (Top) vs. Reconstructed (Bottom)", fontsize=16)
        plt.savefig(os.path.join(save_dir, "VAE_reconstruct.png"))
