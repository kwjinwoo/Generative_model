import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer


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
            return self.data[idx], 0

    class TestDatasetClass:
        def __init__(self, dataset):
            self.loader = DataLoader(dataset, batch_size=2)

    return TestDatasetClass(TestDataset())


@pytest.fixture
def test_trainer():
    trainer_config = {"num_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"}
    return AutoregressiveModelTrainer(trainer_config)


@pytest.fixture
def test_sampler():
    # TODO: Implement Sampler.
    return None
