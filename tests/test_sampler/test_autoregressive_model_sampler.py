import os

from gen_ai.sampler.autoregressive_model_sampler import AutoRegressiveModelSampler


def test_full_sampler(test_model, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    os.makedirs(save_dir)
    num_samples = 4
    sampler = AutoRegressiveModelSampler()

    sampler.full_sample(test_model, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "pixelCNN_generate.png")) is True


def test_half_sample(test_model, test_dataset, tmp_path):
    save_dir = str(tmp_path / "saved_model")
    os.makedirs(save_dir)
    num_samples = 4
    sampler = AutoRegressiveModelSampler()

    sampler.half_sample(test_model, test_dataset, save_dir, num_samples)

    assert os.path.isfile(os.path.join(save_dir, "pixelCNN_half_generate.png")) is True
