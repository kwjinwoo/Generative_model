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
            self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.conv(x)

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
