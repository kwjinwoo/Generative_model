import os

from gen_ai.sampler.autoregressive_model_sampler import AutoRegressiveModelSampler


def test_autregressive_model_sampler(test_model, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    num_samples = 4
    sampler = AutoRegressiveModelSampler()

    sampler.sample(test_model, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "pixelCNN_generate.png")) is True
