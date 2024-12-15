import torch
import torch.nn as nn


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
