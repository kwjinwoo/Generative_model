import os

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

        self.scale_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=in_features),
            nn.Tanh(),
        )
        self.translate_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=in_features),
        )

    def forward(self, inputs: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            return self.inverse_mapping(inputs)
        else:
            return self.forward_mapping(inputs)

    def forward_mapping(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = z * self.mask
        scale = self.scale_layer(x1) * (1 - self.mask)
        translate = self.translate_layer(x1) * (1 - self.mask)

        x = x1 + (z * torch.exp(scale) + translate) * (1 - self.mask)
        return x, scale.sum(dim=1)

    def inverse_mapping(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = x * self.mask
        scale = self.scale_layer(z1) * (1 - self.mask)
        translate = self.translate_layer(z1) * (1 - self.mask)

        z = z1 + ((x - translate) * torch.exp(-scale)) * (1 - self.mask)
        return z, scale.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        masks = [
            torch.tensor(
                [1 if i % 2 == j % 2 else 0 for i in range(28 * 28)],
                dtype=torch.float32,
            )
            for j in range(num_layers)
        ]
        self.layers = nn.ModuleList([AffineCouplingLayer(28 * 28, mask) for mask in masks])

    def forward(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        log_det_jacobian = 0

        if reverse:
            layers = reversed(self.layers)
        else:
            x = x.view(-1, 28 * 28)
            layers = self.layers

        for layer in layers:
            x, log_det = layer(x, reverse)
            log_det_jacobian += log_det
        return x, log_det_jacobian


class NICE(nn.Module):

    def __init__(self, num_layers):
        super().__init__()

        self.m = torch.nn.ModuleList([nn.Sequential(
            nn.Linear(28 * 28 // 2, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000), nn.ReLU(),
            nn.Linear(1000, 1000 // 2), ) for i in range(4)])
        self.s = torch.nn.Parameter(torch.randn(28 * 28))

    def forward(self, x, reverse):
        if reverse:
            x = x.clone() / torch.exp(self.s)
            for i in range(len(self.m) - 1, -1, -1):
                h_i1 = x[:, ::2]
                h_i2 = x[:, 1::2]
                x_i1 = h_i1
                x_i2 = h_i2 - self.m[i](x_i1)
                x = torch.empty(x.shape, device=x.device)
                x[:, ::2] = x_i1 if (i % 2) == 0 else x_i2
                x[:, 1::2] = x_i2 if (i % 2) == 0 else x_i1
            return x
        else:
            x = x.clone()
            x = x.view(-1, 28 * 28)
            for i in range(len(self.m)):
                x_i1 = x[:, ::2] if (i % 2) == 0 else x[:, 1::2]
                x_i2 = x[:, 1::2] if (i % 2) == 0 else x[:, ::2]
                h_i1 = x_i1
                h_i2 = x_i2 + self.m[i](x_i1)
                x = torch.empty(x.shape, device=x.device)
                x[:, ::2] = h_i1
                x[:, 1::2] = h_i2
            z = torch.exp(self.s) * x
            log_jacobian = torch.sum(self.s)
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

    def __init__(self, torch_module: RealNVP, trainer, sampler, dataset):
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        """Train Normalizing Flow Model."""
        self.trainer.criterion = normalizing_flow_loss
        self.trainer.train(self.torch_module, self.dataset.train_loader)

    def sample(self, save_dir: str, num_samples: int) -> None:
        """Sample Normalizing Flow Model."""
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

        torch.save(self.torch_module.state_dict(), os.path.join(save_dir, "RealNVP.pth"))
