import torch.nn as nn
from torch.optim import Adam

from gen_ai.trainer.diffusion_trainer import DiffusionTrainer


def test_diffusion_trainer(test_config, test_model, test_data_loader):
    trainer = DiffusionTrainer(test_config)

    trainer.train(test_model, test_data_loader)

    assert isinstance(trainer.optimizer, Adam)
    assert isinstance(trainer.criterion, nn.MSELoss)
