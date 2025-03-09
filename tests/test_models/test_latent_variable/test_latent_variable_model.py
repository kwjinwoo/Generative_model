from gen_ai.models.latent_variable import LatentVariableModel


def test_laten_variable_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = LatentVariableModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()
