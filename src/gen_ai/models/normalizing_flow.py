import os

import torch
import torch.nn as nn
from torch.distributions import AffineTransform, SigmoidTransform, TransformedDistribution, Uniform

from gen_ai.models import GenAIModelBase


class StandardLogisticDistribution:

    def __init__(self, data_dim: int, device: torch.device):
        self.m = TransformedDistribution(
            Uniform(
                torch.zeros(data_dim, device=device),
                torch.ones(data_dim, device=device),
            ),
            [
                SigmoidTransform().inv,
                AffineTransform(
                    torch.zeros(data_dim, device=device),
                    torch.ones(data_dim, device=device),
                ),
            ],
        )

    def log_prob(self, z):
        return self.m.log_prob(z)

    def sample(self, size: torch.Size | None = None):
        if size is None:
            return self.m.sample()
        else:
            return self.m.sample(size)


class AdditiveCouplingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mask_type: str) -> None:
        super().__init__()
        self.mask_type = mask_type
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_dim // 2, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim // 2),
        )

    def forward(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        x1, x2 = self.chunk(x)

        if reverse:
            return self.inverse_mapping(x1, x2, x.size())
        else:
            return self.forward_mapping(x1, x2, x.size())

    def chunk(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_type == "odd":
            return x[:, 1::2], x[:, 0::2]
        elif self.mask_type == "even":
            return x[:, 0::2], x[:, 1::2]
        else:
            raise ValueError(f"Invalid Mask Type. {self.mask_tpye}")

    def combine(self, x1: torch.Tensor, x2: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = torch.empty(size, device=x1.device)
        if self.mask_type == "odd":
            x[:, 1::2] = x1
            x[:, 0::2] = x2
        elif self.mask_type == "even":
            x[:, 0::2] = x1
            x[:, 1::2] = x2
        else:
            raise ValueError(f"Invalid Mask Type. {self.mask_type}")
        return x

    def forward_mapping(self, x1: torch.Tensor, x2: torch.Tensor, size: torch.Size) -> torch.Tensor:
        h1 = x1
        h2 = x2 + self.layer(x1)
        return self.combine(h1, h2, size)

    def inverse_mapping(self, z1: torch.Tensor, z2: torch.Tensor, size: torch.Size) -> torch.Tensor:
        h1 = z1
        h2 = z2 - self.layer(z1)
        return self.combine(h1, h2, size)


class ScalingLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.randn(input_dim))

    def forward(self, x: torch.Tensor, reverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            return x * torch.exp(-self.scale), -self.scale.sum()
        else:
            return x * torch.exp(self.scale), self.scale.sum()


class NICE(nn.Module):

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()

        self.coupling_layer = nn.ModuleList(
            [AdditiveCouplingLayer(28 * 28, hidden_dim, "odd" if i % 2 == 0 else "even") for i in range(num_layers)]
        )
        self.scaling_layer = ScalingLayer(28 * 28)

    def forward(self, x: torch.Tensor, reverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.clone()
        if reverse:
            return self.inverse_mapping(x)
        else:
            x = x.view(-1, 28 * 28)
            return self.forward_mapping(x)

    def forward_mapping(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.coupling_layer:
            x = layer(x, reverse=False)
        return self.scaling_layer(x, reverse=False)

    def inverse_mapping(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, log_jacobian = self.scaling_layer(z, reverse=True)
        for layer in reversed(self.coupling_layer):
            z = layer(z, reverse=True)
        return z, log_jacobian


def normalizing_flow_loss(
    z: torch.Tensor,
    log_det_jacobian: torch.Tensor,
    prior: torch.distributions.Distribution,
) -> torch.Tensor:
    log_prob = torch.sum(prior.log_prob(z), dim=1)
    return -(log_prob + log_det_jacobian).mean()


class NormalizingFlowModel(GenAIModelBase):
    torch_module_class = NICE

    def __init__(self, torch_module: NICE, trainer, sampler, dataset):
        super().__init__(torch_module, trainer, sampler, dataset)
        self.prior = StandardLogisticDistribution(28 * 28, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def train(self) -> None:
        """Train Normalizing Flow Model."""
        self.trainer.criterion = normalizing_flow_loss
        self.trainer.prior = self.prior
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir: str, num_samples: int) -> None:
        """Sample Normalizing Flow Model."""
        self.sampler.prior = self.prior
        self.sampler.sample(self.torch_module, self.dataset.valid_dataset, save_dir, num_samples)

    def load(self, file_path: str) -> None:
        """Load trained model from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        self.torch_module.load_state_dict(torch.load(file_path, map_location=self.sampler.device))

    def save(self, save_dir: str) -> None:
        """save trained model to save dir.

        Args:
            save_dir (str): save directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.torch_module.state_dict(), os.path.join(save_dir, "NICE.pth"))
