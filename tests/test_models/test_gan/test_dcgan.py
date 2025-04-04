import torch

from gen_ai.models.generative_adversarial_network import DCGAN, Discriminator, Generator


def test_generator():
    noise_dim = 16
    generator = Generator(noise_dim)

    temp_inp = torch.randn(2, noise_dim)

    output = generator(temp_inp)
    assert output.size() == (2, 1, 28, 28)


def test_discriminator():
    discriminator = Discriminator()

    temp_inp = torch.randn(2, 1, 28, 28)

    output = discriminator(temp_inp)
    assert output.size() == (2, 1)


def test_dcgan():
    noise_dim = 16
    dcgan = DCGAN(noise_dim)

    temp_inp = torch.randn(2, 1, 28, 28)

    fake_output, real_output = dcgan(temp_inp, generator_only=False)
    assert fake_output.size() == (2, 1)
    assert real_output.size() == (2, 1)

    fake_output_only = dcgan(temp_inp, generator_only=True)
    assert fake_output_only.size() == (2, 1)
