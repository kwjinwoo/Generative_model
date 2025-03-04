import pytest
import torch.nn as nn


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
