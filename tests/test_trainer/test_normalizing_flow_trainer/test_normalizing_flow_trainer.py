from torch.optim import Adam

from gen_ai.models.normalizing_flow import StandardLogisticDistribution, normalizing_flow_loss
from gen_ai.trainer.normalizing_flow_model_trainer import NormalizingFlowModelTrainer


def test_normalizing_flow_model_trainer(test_config, test_model, test_data_loader):
    trainer = NormalizingFlowModelTrainer(test_config)
    trainer.criterion = normalizing_flow_loss
    trainer.prior = StandardLogisticDistribution(28 * 28, "cpu")

    trainer.train(test_model, test_data_loader)

    assert isinstance(trainer.optimizer, Adam)
    assert trainer.criterion == normalizing_flow_loss
