import pytest
import torch.nn as nn


@pytest.fixture
def test_model():
    class Generator(nn.Moudle):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = x.view(-1, 1, 8, 8)
            return self.layer(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x = self.layer(x)
            return x.view(-1, 28 * 28, 1)

    class GAN(nn.Module):
        def __init__(self):
            super().__init__()
            self.generator = Generator()
            self.discriminator = Discriminator()

        def generator_forward(self, inputs):
            return self.generator(inputs)

        def discriminator_forward(self, inputs):
            return self.discriminator(inputs)

    return GAN()
