import pytest
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from gen_ai.dataset import MNISTLoader


@pytest.mark.skip(reason="Downloading MNIST dataset is slow.")
def test_mnist_loader(test_data_config, tmpdir) -> None:
    mnist_loader = MNISTLoader(config=test_data_config, path=tmpdir)

    assert isinstance(mnist_loader.dataset, MNIST)
    assert isinstance(mnist_loader.loader, DataLoader)
    assert len(mnist_loader.loader) == 1875
    assert len(mnist_loader.loader.dataset) == 60000
    assert mnist_loader.loader.batch_size == 32
    assert mnist_loader.loader.num_workers == 4
