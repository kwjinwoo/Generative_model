import os

from gen_ai.models.autoregressive import AutoregressiveModel


def test_autoregressive_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = AutoregressiveModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()


def test_autoregressive_model_sample(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_dir = str(tmp_path / "assets")
    num_samples = 4
    model = AutoregressiveModel(test_model, test_trainer, test_sampler, test_dataset)
    model.sample(save_dir, num_samples)


def test_autoregressive_model_save(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_path = str(tmp_path)
    model = AutoregressiveModel(test_model, test_trainer, test_sampler, test_dataset)
    model.save(save_path)

    assert os.path.isfile(os.path.join(save_path, "PixelCNN.pth"))
