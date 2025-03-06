from __future__ import annotations

import os
import typing

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase

if typing.TYPE_CHECKING:
    from gen_ai.dataset import MNISTDataset
    from gen_ai.sampler.autoregressive_model_sampler import AutoRegressiveModelSampler
    from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer


class MaskedConv2D(nn.Conv2d):
    """Applies Masked Convolution over an unputs.
    if mask type is 'A', center of features is not maked.
    if mask type is 'B', center of feature is maked.

    Args:
        mask_type (str): mask type of Masked Convolution.

    Raises:
        ValueError: if mask type is invalid, raise ValueError.
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight.data.clone()))

        yc, xc = self.weight.data.size()[-2] // 2, self.weight.data.size()[-1] // 2
        self.mask[..., :yc, :] = 1.0
        self.mask[..., yc, : xc + 1] = 1.0

        if mask_type == "A":
            self.mask[..., yc, xc] = 0.0
        elif mask_type == "B":
            ...
        else:
            raise ValueError(f"Mask Type {mask_type} is Invalid Value. Mask Type Must be in [A or B]")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """apply maked conv to inputs.

        Args:
            inputs (torch.Tensor): inputs tensors.

        Returns:
            torch.Tensor: output of masked conv.
        """
        self.weight.data *= self.mask
        return super().forward(inputs)


class MaskedConvResidualBlock(nn.Module):
    """Masked Conv's Residual Block.
    it consist of 2 conv and 1 masked conv with relu activation.

    Args:
        in_channels (int): the number of input channels.
    """

    def __init__(self, in_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            MaskedConv2D(
                mask_type="B",
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply MaskedConvResidualBlock to input tensor x.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: MaskedConvResidualBlock's output.
        """
        out = self.layers(x)
        return out + x


class PixelCNN(nn.Module):
    """Pixel CNN module.

    Args:
        num_channels (int): the number of channels used at all conv modules.
        num_layers (int): the nuber of residual blocks.
    """

    def __init__(self, num_channels: int, num_layers: int, img_channel: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_layer = nn.Sequential(
            MaskedConv2D(
                mask_type="A",
                in_channels=img_channel,
                out_channels=num_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.layers = nn.Sequential(
            *[MaskedConvResidualBlock(in_channels=num_channels, kernel_size=7) for _ in range(num_layers)]
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """feed fowrading pixel cnn.

        Args:
            inputs (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: pixel cnn's output.
        """
        x = self.input_layer(inputs)
        x = self.layers(x)
        out = self.out_layer(x)
        return out


class AutoregressiveModel(GenAIModelBase):
    torch_module_class = PixelCNN

    def __init__(
        self,
        torch_module: PixelCNN,
        trainer: AutoregressiveModelTrainer,
        sampler: AutoRegressiveModelSampler,
        dataset: MNISTDataset,
    ) -> None:
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir: str, num_samples: int) -> None:
        self.sampler.sample(self.torch_module, self.dataset.valid_dataset, save_dir, num_samples)

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        self.torch_module.load_state_dict(torch.load(file_path, map_location=self.sampler.device, weights_only=True))

    def save(self, save_dir: str):
        """save trained model to save dir.

        Args:
            save_dir (str): svae directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.torch_module.state_dict(),
            os.path.join(save_dir, "autoregressive_model.pth"),
        )
