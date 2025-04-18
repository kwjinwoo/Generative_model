import os

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


class Discriminator(nn.Module):
    """Discriminator for DCGAN."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifer = nn.Sequential(nn.Linear(in_features=7 * 7 * 128, out_features=1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layers(inputs)
        x = x.view(-1, 7 * 7 * 128)
        return self.classifer(x)


class Generator(nn.Module):
    """Generator for DCGAN."""

    def __init__(self, noise_dim: int) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=noise_dim, out_features=7 * 7 * 256, bias=False),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.ReLU(),
        )
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(inputs)
        x = x.view(-1, 256, 7, 7)
        return self.layer(x)


class DCGAN(nn.Module):
    """DCGAN model."""

    def __init__(self, noise_dim: int) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim)
        self.discriminator = Discriminator()

    def forward(self, input: torch.Tensor, generator_only: bool) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """forward of DCGAN.

        Args:
            input (torch.Tensor): input tensor.
            generator_only (bool): if True, return only fake output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: fake output and real output
        """
        noise = torch.randn(input.size(0), self.noise_dim, device=input.device)
        generated_image = self.generator(noise)
        fake_output = self.discriminator(generated_image)
        if generator_only:
            return fake_output
        real_output = self.discriminator(input)
        return fake_output, real_output


class GenerativeAdversarialNetworkModel(GenAIModelBase):
    """Generative Adversarial Network model."""

    torch_module_class = DCGAN

    def __init__(self, torch_model, trainer, sampler, dataset):
        super().__init__(torch_model, trainer, sampler, dataset)

    def train(self) -> None:
        """Train the model."""
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir: str, num_samples: int) -> None:
        """Sample the model."""
        self.sampler.sample(self.torch_module, save_dir, num_samples)

    def load(self, file_path: str) -> None:
        """Load the model.

        Args:
            file_path (str): path to the model file.

        Raises:
            FileNotFoundError: if the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.torch_module.load_state_dict(torch.load(file_path))

    def save(self, save_dir: str) -> None:
        """Save the model.

        Args:
            save_dir (str): save directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.torch_module.state_dict(),
            os.path.join(save_dir, f"{self.torch_module_class.__name__}.pth"),
        )
        print(os.path.join(save_dir, f"{self.torch_module_class.__name__}.pth"))
