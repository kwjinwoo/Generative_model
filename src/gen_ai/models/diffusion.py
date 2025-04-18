import os

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward of SinusoidalTimeEmbedding.

        Args:
            t (torch.Tensor): time tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: sinusoidal time embedding of shape (batch_size, dim).
        """
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConvBlock(nn.Module):
    """Convolutional block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, kernel_size=3) -> None:
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Foward of ConvBlock.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_channels, height, width).
            t_emb (torch.Tensor): time embedding of shape (batch_size, time_emb_dim).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_channels, height, width).
        """
        h = self.conv(x)
        t = self.time_proj(t_emb)[:, :, None, None]
        return h + t


class Downsample(nn.Module):
    """Downsample block."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Upsample(nn.Module):
    """Upsample block."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet(nn.Module):
    """UNet model for diffusion."""

    def __init__(self, diffusion_step: int, time_emb_dim: int):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.register_buffer("betas", torch.linspace(0.0001, 0.02, diffusion_step).float())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_hat", torch.cumprod(self.alphas, dim=0))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.enc1 = ConvBlock(1, 32, time_emb_dim)
        self.down1 = Downsample(32)
        self.enc2 = ConvBlock(32, 64, time_emb_dim)
        self.down2 = Downsample(64)
        self.bot = ConvBlock(64, 128, time_emb_dim)
        self.up2 = Upsample(128)
        self.dec2 = ConvBlock(128 + 64, 64, time_emb_dim)
        self.up1 = Upsample(64)
        self.dec1 = ConvBlock(64 + 32, 32, time_emb_dim)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward of UNet.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, height, width).
            t (torch.Tensor): time tensor of shape (batch_size, 1).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, 1, height, width).
        """
        t_emb = self.time_mlp(t)

        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        xb = self.bot(self.down2(x2), t_emb)

        xd2 = self.dec2(torch.cat([self.up2(xb), x2], dim=1), t_emb)
        xd1 = self.dec1(torch.cat([self.up1(xd2), x1], dim=1), t_emb)

        return self.out(xd1)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise=None):
        """Disturb x_0 to x_t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_hat = torch.sqrt(self.alphas_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alphas_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise


class DiffusionModel(GenAIModelBase):
    """Diffusion model class."""

    torch_module_class = UNet

    def __init__(self, torch_module, trainer, sampler, dataset) -> None:
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        """train the model."""
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir, num_samples):
        """sample the model."""
        self.sampler.sample(self.torch_module, save_dir, num_samples)

    def load(self, file_path: str) -> None:
        """load trained model from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.torch_module.load_state_dict(torch.load(file_path, map_location=self.sampler.device, weights_only=True))

    def save(self, save_dir: str) -> None:
        """save trained model to save dir.

        Args:
            save_dir (str): svae directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.torch_module.state_dict(),
            os.path.join(save_dir, f"{self.torch_module_class.__name__}.pth"),
        )
        print(os.path.join(save_dir, f"{self.torch_module_class.__name__}.pth"))
