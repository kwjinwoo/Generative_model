from enum import StrEnum


class ModelType(StrEnum):
    autoregressive = "autoregressive"
    latent_variable = "latent_variable"
    normalizing_flow = "normalizing_flow"
