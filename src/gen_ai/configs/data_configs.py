from dataclasses import dataclass


@dataclass
class DataConfig:
    data_type: str
    batch_size: int
    num_workers: int = 2
