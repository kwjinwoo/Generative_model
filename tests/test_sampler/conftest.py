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
            self.linear = nn.Linear(in_features=10, out_features=28 * 28)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.linear(x)
            out = self.sigmoid(out)
            return out.view(-1, 1, 28, 28)

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
            self.flatten = nn.Flatten()
            self.layer = nn.Linear(28 * 28, 28 * 28)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, reverse):
            if reverse:
                x = self.flatten(x)
                x = self.layer(x)
                return self.sigmoid(x.view(-1, 1, 28, 28)), torch.sum(x, dim=1)
            else:
                return self.sigmoid(x.view(-1, 1, 28, 28)), torch.sum(x, dim=1)

    return TestModel()
