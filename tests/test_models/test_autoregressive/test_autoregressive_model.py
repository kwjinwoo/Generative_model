from gen_ai.models.autoregressive import AutoregressiveModel


def test_autoregressive_model(test_model, test_trainer, test_sampler, test_dataset):
    model = AutoregressiveModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()
