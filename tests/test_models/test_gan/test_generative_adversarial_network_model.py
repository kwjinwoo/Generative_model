from gen_ai.models.generative_adversarial_network import GenerativeAdversarialNetworkModel


def test_generative_adversarial_network_model_train(test_model, test_trainer, test_sampler, test_dataset):
    model = GenerativeAdversarialNetworkModel(test_model, test_trainer, test_sampler, test_dataset)
    model.train()


def test_autoregressive_model_sample(test_model, test_trainer, test_sampler, test_dataset, tmp_path):
    save_dir = str(tmp_path / "assets")
    num_samples = 4
    model = GenerativeAdversarialNetworkModel(test_model, test_trainer, test_sampler, test_dataset)
    model.sample(save_dir, num_samples)
