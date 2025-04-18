import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class DiffusionSampler:
    """Diffusion Sampler for diffusion models."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, model: nn.Module, saved_dir: str, num_samples: int) -> None:
        """sample method to sample from diffusion models.

        Args:
            model (nn.Module): diffusion model
            saved_dir (str): saved directory
        """
        if os.path.exists(saved_dir) is False:
            os.makedirs(saved_dir)

        model.eval()
        model.to(self.device)
        self.full_sample(model, saved_dir, num_samples)

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, num_samples: int):
        """sample from diffusion model."""

        diffusion_step = model.diffusion_step
        x = torch.randn([num_samples, 1, 28, 28]).to(self.device)
        for t_ in reversed(range(diffusion_step)):
            t = torch.full((num_samples,), t_, device=self.device, dtype=torch.long)
            noise_pred = model(x, t)
            beta_t = model.betas[t_]
            alpha_t = model.alphas[t_]
            alpha_hat_t = model.alphas_hat[t_]
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1 - alpha_hat_t)
            x = coef1 * (x - coef2 * noise_pred)
            if t_ > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = x + sigma * noise
        return x

    def full_sample(self, model: nn.Module, saved_dir: str, num_samples: int) -> None:
        """sample full image from diffusion model."""
        generated = self.p_sample_loop(model, num_samples)
        generated = generated.cpu()
        num_cols = math.sqrt(num_samples)
        if not num_cols.is_integer():
            raise ValueError("num_samples must be a square number.")
        for i in range(num_samples):
            plt.subplot(int(num_cols), int(num_cols), i + 1)
            plt.imshow(generated[i].permute(1, 2, 0).numpy(), cmap="gray")
            plt.axis("off")
        plt.suptitle("Diffusion generated samples")
        plt.savefig(os.path.join(saved_dir, "diffusion_generate.png"))
        print(f"Diffusion Full Sampling Finished. saved at {saved_dir}")
