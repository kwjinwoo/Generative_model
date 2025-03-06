import pytest
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from gen_ai.dataset import MNISTDataset


@pytest.mark.skip(reason="Downloading MNIST dataset is slow.")
def test_mnist_loader(test_data_config, tmpdir) -> None:
    mnist_loader = MNISTDataset(config=test_data_config, path=tmpdir)

    assert isinstance(mnist_loader.train_dataset, MNIST)
    assert isinstance(mnist_loader.train_loader, DataLoader)
    assert len(mnist_loader.train_loader) == 1875
    assert len(mnist_loader.train_loader.dataset) == 60000
    assert mnist_loader.train_loader.batch_size == 32
    assert mnist_loader.train_loader.num_workers == 4

    assert isinstance(mnist_loader.valid_dataset, MNIST)
    assert isinstance(mnist_loader.valid_loader, DataLoader)
    assert len(mnist_loader.valid_loader) == 313
    assert len(mnist_loader.valid_loader.dataset) == 10000
    assert mnist_loader.valid_loader.batch_size == 32
    assert mnist_loader.valid_loader.num_workers == 4
