import torch.nn as nn
from torch.optim import Adam

from gen_ai.trainer.autoregressive_model_trainer import AutoregressiveModelTrainer


def test_autoregressive_trainer(test_config, test_model, test_data_loader):
    trainer = AutoregressiveModelTrainer(test_config)

    trainer.train(test_model, test_data_loader)

    assert isinstance(trainer.optimizer, Adam)
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
