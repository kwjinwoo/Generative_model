import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@pytest.fixture
def test_config():
    config = {"num_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"}
    return config


@pytest.fixture
def test_model():
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
def test_data_loader():
    class TestDataset(Dataset):
        def __init__(self):
            self.data = torch.rand(10, 1, 28, 28)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], 0

    return DataLoader(TestDataset(), batch_size=2)
