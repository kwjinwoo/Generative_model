from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str
    img_channel: int


@dataclass
class AutoregressiveConfig(ModelConfig):
    num_channels: int
    num_layers: int
