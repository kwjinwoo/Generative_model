from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_epochs: int
    learning_rate: float = 0.001
    optimizer: str = "adam"
