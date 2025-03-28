import pytest
import torch

from gen_ai.models.autoregressive import MaskedConv2D, MaskedConvResidualBlock, PixelCNN


@pytest.mark.parametrize(
    "mask_type, answer_mask", [("A", [[1, 1, 1], [1, 0, 0], [0, 0, 0]]), ("B", [[1, 1, 1], [1, 1, 0], [0, 0, 0]])]
)
def test_masked_conv_2d(mask_type, answer_mask):
    masked_conv = MaskedConv2D(mask_type, in_channels=3, out_channels=3, kernel_size=3, padding=1)
    answer_mask = torch.tensor(answer_mask).reshape(1, 1, 3, 3)

    mask: torch.Tensor = masked_conv.mask

    assert torch.Size([3, 3, 3, 3]) == mask.size()
    assert torch.all(mask == answer_mask)


def test_masked_conv_residual_block():
    block = MaskedConvResidualBlock(3, 3)

    tmp_input = torch.randn(1, 3, 10, 10)

    block_out = block(tmp_input)

    assert block_out.size() == torch.Size([1, 3, 10, 10])


def test_pixel_cnn():
    pixel_cnn = PixelCNN(120, 5, 1)

    tmp_input = torch.randn(1, 1, 10, 10)

    out = pixel_cnn(tmp_input)

    assert out.size() == torch.Size([1, 256, 10, 10])
