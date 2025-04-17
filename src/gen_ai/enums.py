from enum import StrEnum


class ModelType(StrEnum):
    autoregressive = "autoregressive"
    latent_variable = "latent_variable"
    normalizing_flow = "normalizing_flow"
    generative_adversarial_network = "generative_adversarial_network"
    diffusion = "diffusion"
