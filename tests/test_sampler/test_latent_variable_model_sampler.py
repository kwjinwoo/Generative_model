import os

from gen_ai.sampler.latent_variable_model_sampler import LatentVariableModelSampler


def test_sample(test_latent_model, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    os.makedirs(save_dir)
    num_samples = 4
    sampler = LatentVariableModelSampler()

    sampler.sample(test_latent_model, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "VAE_generate.png")) is True
