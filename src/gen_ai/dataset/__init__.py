from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from gen_ai.configs import GenAIConfig


class MNISTLoader:
    """MNIST DataLoader Class."""

    def __init__(self, config: GenAIConfig, path: str = "./data") -> None:
        """MNISTLoader Constructor.

        Args:
            config (GenAIConfig): config.
            path (str, optional): path to save data. Defaults to "./data".
        """
        self.config = config
        self.path = path
        self.dataset = self._get_dataset()

        self._loader = self._make_loader()

    @property
    def loader(self) -> DataLoader:
        """loader property.

        Returns:
            DataLoader: loader.
        """
        if self._loader is None:
            self._loader = self._make_loader()
        return self._loader

    def _make_mnist_transform(self) -> Compose:
        """make mnist transform.

        Returns:
            Compose: mnist transform.
        """
        return Compose([ToTensor()])

    def _get_dataset(self) -> MNIST:
        """get MNIST dataset.

        Returns:
            MNIST: MNIST dataset.
        """
        return MNIST(
            root=self.path,
            train=True,
            download=True,
            transform=self._make_mnist_transform(),
        )

    def _make_loader(self) -> DataLoader:
        """make loader.

        Returns:
            DataLoader: loader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config.data_config["batch_size"],
            shuffle=True,
            num_workers=self.config.data_config.get("num_workers", 2),
        )


__all__ = ["MNISTLoader"]
