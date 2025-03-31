import os

from gen_ai.models.latent_variable import LatentVariableModel


def test_laten_variable_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = LatentVariableModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()


def test_latent_variable_model_sample(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_dir = str(tmp_path / "assets")
    num_samples = 4
    model = LatentVariableModel(test_model, test_trainer, test_sampler, test_dataset)
    model.sample(save_dir, num_samples)


def test_autoregressive_model_save(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_path = str(tmp_path)
    model = LatentVariableModel(test_model, test_trainer, test_sampler, test_dataset)
    model.save(save_path)

    assert os.path.isfile(os.path.join(save_path, "ConvolutionalVAE.pth"))
