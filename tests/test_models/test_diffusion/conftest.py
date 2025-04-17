import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gen_ai.sampler.diffusion_sampler import DiffusionSampler
from gen_ai.trainer.diffusion_trainer import DiffusionTrainer


@pytest.fixture
def test_model():
    class TestDiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("betas", torch.linspace(0.0001, 0.02, 1000))
            self.register_buffer("alphas", 1.0 - self.betas)
            self.register_buffer("alphas_hat", torch.cumprod(self.alphas, dim=0))
            self.diffusion_step = 10
            self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        def forward(self, x, t):
            return self.conv(x)

        def q_sample(self, x, t, noise):
            return x

    return TestDiffusionModel()


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
    trainer_config = {"num_epochs": 1, "learning_rate": 0.001, "optimizer": "adam"}
    return DiffusionTrainer(trainer_config)


@pytest.fixture
def test_sampler():
    return DiffusionSampler()
