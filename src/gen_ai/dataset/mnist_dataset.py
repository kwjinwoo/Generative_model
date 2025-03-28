from typing import Optional

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataset:
    """MNIST Dataset Class."""

    def __init__(self, config: dict[str, int], path: Optional[str] = "./data") -> None:
        """MNISTLoader Constructor.

        Args:
            config (dict[str, int]): config.
            path (str, optional): path to save data. Defaults to "./data".
        """
        self.config = config
        self.path = path
        self.train_dataset = self._get_dataset(train=True)
        self.valid_dataset = self._get_dataset(train=False)

        self._train_loader = self._make_loader(train=True)
        self._valid_loader = self._make_loader(train=False)

    @property
    def train_loader(self) -> DataLoader:
        """train loader property.

        Returns:
            DataLoader: loader.
        """
        if self._train_loader is None:
            self._train_loader = self._make_loader(train=True)
        return self._train_loader

    @property
    def valid_loader(self) -> DataLoader:
        """valid loader property.

        Returns:
            DataLoader: loader.
        """
        if self._valid_loader is None:
            self._valid_loader = self._make_loader(train=False)
        return self._valid_loader

    def _get_dataset(self, train: bool) -> MNIST:
        """get MNIST dataset.

        Args:
            train (bool): train or test.
        Returns:
            MNIST: MNIST dataset.
        """
        return MNIST(root=self.path, train=train, download=True, transform=ToTensor())

    def _make_loader(self, train: bool) -> DataLoader:
        """make loader.

        Args:
            train (bool): train or test.

        Returns:
            DataLoader: loader.
        """
        return DataLoader(
            dataset=self.train_dataset if train else self.valid_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
