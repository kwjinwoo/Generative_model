from gen_ai.models.diffusion import DiffusionModel


def test_diffusion_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = DiffusionModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()


def test_diffusion_model_sample(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_dir = str(tmp_path / "assets")
    num_samples = 4
    model = DiffusionModel(test_model, test_trainer, test_sampler, test_dataset)
    model.sample(save_dir, num_samples)
