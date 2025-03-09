from torch.optim import Adam

from gen_ai.models.latent_variable import elbo_loss
from gen_ai.trainer.latent_variable_model_trainer import LatentVariableModelTrainer


def test_latent_variable_trainer(test_config, test_model, test_data_loader):
    trainer = LatentVariableModelTrainer(test_config)
    trainer.criterion = elbo_loss

    trainer.train(test_model, test_data_loader)

    assert isinstance(trainer.optimizer, Adam)
    assert trainer.criterion == elbo_loss
