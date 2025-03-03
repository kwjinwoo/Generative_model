import os

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


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

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            MaskedConv2D(
                mask_type="B",
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
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
        self.layers = nn.Sequential(*[MaskedConvResidualBlock(in_channels=num_channels) for _ in range(num_layers)])
        self.last_conv = nn.Sequential(
            MaskedConv2D(
                mask_type="B",
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            MaskedConv2D(
                mask_type="B",
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
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
        x = self.last_conv(x)
        out = self.out_layer(x)
        return out


class AutoregressiveModel(GenAIModelBase):
    torch_module_class = PixelCNN

    def __init__(self, torch_module, trainer, sampler, dataset) -> None:
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        self.trainer.train(self.torch_module, self.dataset.loader)

    def sample(self) -> None:
        self.sampler.sample()

    # TODO: Implement load method
    def load(self, file_path):
        pass

    def save(self, save_dir: str):
        """save trained model to save dir.

        Args:
            save_dir (str): svae directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.torch_module.state_dict(), os.path.join(save_dir, "autoregressive_model.pth"))
