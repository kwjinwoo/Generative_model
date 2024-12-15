import pytest
import torch

from gen_ai.models.autoregressive import MaskedConv2D


@pytest.mark.parametrize(
    "mask_type, answer_mask", [("A", [[1, 1, 1], [1, 0, 0], [0, 0, 0]]), ("B", [[1, 1, 1], [1, 1, 0], [0, 0, 0]])]
)
def test_maskedConv2D(mask_type, answer_mask):
    masked_conv = MaskedConv2D(mask_type, in_channels=3, out_channels=3, kernel_size=3, padding=1)
    answer_mask = torch.tensor(answer_mask).reshape(1, 1, 3, 3)

    mask: torch.Tensor = masked_conv.mask

    assert torch.Size([3, 3, 3, 3]) == mask.size()
    assert torch.all(mask == answer_mask)
