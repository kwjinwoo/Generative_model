import os

from gen_ai.sampler.generative_adversarial_network_sampler import GenerativeAdversarialNetworkSampler


def test_random_sample(test_gan_model, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    os.makedirs(save_dir)
    num_samples = 4
    sampler = GenerativeAdversarialNetworkSampler()

    sampler.random_sample(test_gan_model, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "GAN_generated.png")) is True
