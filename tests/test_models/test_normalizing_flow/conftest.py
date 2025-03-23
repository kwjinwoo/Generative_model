import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gen_ai.sampler.normalizing_flow_model_sampler import NormalizingFlowModelSampler
from gen_ai.trainer.normalizing_flow_model_trainer import NormalizingFlowModelTrainer


@pytest.fixture
def test_model():
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.flatten = nn.Flatten()
            self.layer = nn.Linear(28 * 28, 28 * 28)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, reverse):
            if reverse:
                x = self.layer(x)
                return self.sigmoid(x), torch.sum(x, dim=1)
            else:
                x = x.view(-1, 28 * 28)
                x = self.layer(x)
                return self.sigmoid(x), torch.sum(x, dim=1)

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
            self.valid_dataset = dataset
            self.train_loader = DataLoader(dataset, batch_size=2)

    return TestDatasetClass(TestDataset())


@pytest.fixture
def test_trainer():
    trainer_config = {}
    trainer_config = {"num_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"}
    return NormalizingFlowModelTrainer(trainer_config)


@pytest.fixture
def test_sampler():
    return NormalizingFlowModelSampler()
