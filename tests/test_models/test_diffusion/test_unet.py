import torch

from gen_ai.models.diffusion import UNet


def test_unet():
    unet = UNet(50, 16)

    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 50, (2,))

    disturbed = unet.q_sample(x, t)
    assert disturbed.size() == torch.Size([2, 1, 28, 28])

    out = unet(x, t)
    assert out.size() == torch.Size([2, 1, 28, 28])
