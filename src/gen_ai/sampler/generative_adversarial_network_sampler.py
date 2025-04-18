import math
import os

import matplotlib.pyplot as plt
import torch


class GenerativeAdversarialNetworkSampler:
    """ "Generative Adversarial Network Sampler for GAN models."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model, save_dir: str, num_samples: int) -> None:
        """sample method to sample from GAN models.

        Args:
            model (nn.Module): GAN model.
            save_dir (str): saved directory.
            num_samples (int): the number of samples.
        """
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        model.eval()
        model.to(self.device)

        self.random_sample(model, save_dir, num_samples)

    def random_sample(self, model, save_dir: str, num_samples: int) -> None:
        """random_sample method to sample from GAN models.

        Args:
            model (nn.Module): GAN model.
            save_dir (str): saved directory.
            num_samples (int): the number of samples.
        """
        print("GAN Model Radom Sampling Start.")
        with torch.no_grad():
            noise = torch.randn(num_samples, model.noise_dim).to(self.device)
            generated = model.generator(noise)

        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")

        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("GAN generated samples")
        plt.savefig(os.path.join(save_dir, "DCGAN_generated.png"))
        print(f"GAN Finished. saved at {save_dir}")
