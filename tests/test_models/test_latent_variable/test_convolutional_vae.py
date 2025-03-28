import torch

from gen_ai.models.latent_variable import ConvolutionalVAE


def test_convolutional_vae():
    vae = ConvolutionalVAE(1, 4)

    tmp_inp = torch.randn(4, 1, 28, 28)

    out, mean, log_var = vae(tmp_inp)

    assert out.size() == torch.Size([4, 256, 28, 28])
    assert mean.size() == torch.Size([4, 4])
    assert log_var.size() == torch.Size([4, 4])
