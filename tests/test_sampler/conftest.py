import random

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@pytest.fixture
def test_model():
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.conv(x))

    return TestModel()


@pytest.fixture
def test_dataset():
    class TestDataset(Dataset):
        def __init__(self):
            self.data = torch.rand(10, 1, 28, 28)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], random.randint(0, 10)

    return TestDataset()


@pytest.fixture
def test_latent_model():
    class TestEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 10
            self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            return self.layers(x), self.layers(x)

    class TestDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = 10
            self.linear = nn.Linear(in_features=10, out_features=256 * 28 * 28)

        def forward(self, x):
            out = self.linear(x)
            return out.view(-1, 256, 28, 28)

    class TestLatentModel(nn.Module):
        def __init__(self):
            self.latent_dim = 10
            super().__init__()
            self.encoder = TestEncoder()
            self.decoder = TestDecoder()

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    return TestLatentModel()


@pytest.fixture
def test_normalizing_model():
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.layer = nn.Linear(28 * 28, 28 * 28)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, reverse):
            if reverse:
                x = x.view(-1, 28 * 28)
                x = self.layer(x)
                return self.sigmoid(x), torch.sum(torch.randn(28 * 28))
            else:
                return self.sigmoid(x), torch.sum(torch.randn(28 * 28))

    return TestModel()


@pytest.fixture
def test_gan_model():
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(in_features=16, out_features=1 * 28 * 28)

        def forward(self, x):
            x = self.layer(x)
            return x.view(-1, 1, 28, 28)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.layer(x)
            return torch.sigmoid(x.view(-1, 28 * 28, 1))

    class GAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_dim = 16
            self.generator = Generator()
            self.discriminator = Discriminator()

        def forward(self, x, generator_only):
            noise = torch.randn(x.size(0), self.noise_dim).to(x.device)
            generated_output = self.generator(noise)
            fake_output = self.discriminator(generated_output)
            if generator_only:
                return fake_output
            real_output = self.discriminator(x)
            return fake_output, real_output

    return GAN()


@pytest.fixture
def test_diffusion_model():
    class DiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("betas", torch.linspace(0.0001, 0.02, 1000))
            self.register_buffer("alphas", 1.0 - self.betas)
            self.register_buffer("alphas_hat", torch.cumprod(self.alphas, dim=0))
            self.diffusion_step = 10
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        def forward(self, x, t):
            return self.conv(x)

    return DiffusionModel()
