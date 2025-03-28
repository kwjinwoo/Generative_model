import random

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gen_ai.sampler.latent_variable_model_sampler import LatentVariableModelSampler
from gen_ai.trainer.latent_variable_model_trainer import LatentVariableModelTrainer


@pytest.fixture
def test_model():
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

    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.latent_dim = 10
            self.encoder = TestEncoder()
            self.decoder = TestDecoder()
            self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x), torch.rand(1), torch.rand(1)

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

    class TestDatasetClass:
        def __init__(self, dataset):
            self.valid_dataset = dataset
            self.train_loader = DataLoader(dataset, batch_size=2)

    return TestDatasetClass(TestDataset())


@pytest.fixture
def test_trainer():
    trainer_config = {"num_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"}
    return LatentVariableModelTrainer(trainer_config)


@pytest.fixture
def test_sampler():
    return LatentVariableModelSampler()
