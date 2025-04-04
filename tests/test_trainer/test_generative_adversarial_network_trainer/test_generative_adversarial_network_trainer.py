import torch.nn as nn
from torch.optim import Adam

from gen_ai.trainer.generative_adversarial_network_trainer import GenerativeAdversarialNetworkTrainer


def test_adversarial_gan_model_trainer(test_config, test_model, test_data_loader):
    trainer = GenerativeAdversarialNetworkTrainer(test_config)
    trainer.train(test_model, test_data_loader)

    assert isinstance(trainer.get_optimizer("generator"), Adam)
    assert isinstance(trainer.get_optimizer("discriminator"), Adam)
    assert isinstance(trainer.criterion, nn.BCELoss)
