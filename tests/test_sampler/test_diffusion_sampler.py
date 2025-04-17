import os

from gen_ai.sampler.diffusion_sampler import DiffusionSampler


def test_full_smpale(test_diffusion_model, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    os.makedirs(save_dir)
    num_samples = 4
    sampler = DiffusionSampler()

    sampler.sample(test_diffusion_model, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "diffusion_generate.png")) is True
