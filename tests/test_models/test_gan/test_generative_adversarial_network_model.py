from gen_ai.models.generative_adversarial_network import GenerativeAdversarialNetworkModel


def test_generative_adversarial_network_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = GenerativeAdversarialNetworkModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()
