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
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.flatten = nn.Flatten()
            self.layer = nn.Linear(28 * 28, 28 * 28)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.flatten(x)
            x = self.layer(x)
            return self.sigmoid(x.view(-1, 1, 28, 28))

        def inverse(self, x):
            return x.view(-1, 1, 28, 28)

    return TestModel()


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
