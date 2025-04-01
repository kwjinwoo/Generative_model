import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


class Discriminator(nn.Module):
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
        self.classifer = nn.Linear(in_features=7 * 7 * 128, out_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layers(inputs)
        x = x.view(-1, 7 * 7 * 128)
        return self.classifer(x)


class Generator(nn.Module):
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
    def __init__(self, noise_dim: int) -> None:
        super().__init__()
        self.generator = Generator(noise_dim)
        self.discriminator = Discriminator()

    def generator_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.generator(inputs)

    def discriminator_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.discriminator(inputs)


class GenerativeAdversarialNetworkModel(GenAIModelBase):
    torch_module_class = DCGAN

    def __init__(self, torch_model, trainer, sampler, dataset):
        super().__init__(torch_model, trainer, sampler, dataset)

    def train(self) -> None:
        self.trainer.train(self.torch_module, self.dataset.train_loader)
