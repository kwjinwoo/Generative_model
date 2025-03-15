import os

import torch
import torch.nn as nn

from gen_ai.models import GenAIModelBase


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.scale_layer = nn.Sequential(
            nn.Linear(in_features=in_features // 2, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=in_features // 2),
        )
        self.translate_layer = nn.Sequential(
            nn.Linear(in_features=in_features // 2, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=in_features // 2),
        )

    def foward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = inputs.chunk(2, dim=1)
        scale = self.scale_layer(x1).tanh()
        translate = self.translate_layer(x1)

        x2 = x2 * torch.exp(scale) + translate
        return torch.cat([x1, x2], dim=1), scale.sum(dim=1)

    def inverse(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = inputs.chunk(2, dim=1)
        scale = self.scale_layer(x1).tanh()
        translate = self.translate_layer(x1)

        x2 = (x2 - translate) / torch.exp(scale)
        return torch.cat([x1, x2], dim=1), scale.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([AffineCouplingLayer(28 * 28) for _ in range(num_layers)])

    def foward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_jacobian = 0
        x = self.flatten(inputs)

        for layer in self.layers:
            x, log_det = layer(x)
            log_det_jacobian += log_det
        return x, log_det_jacobian

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in reversed(self.layers):
            x, _ = layer(x)
        return x


def normalizing_flow_loss(z: torch.Tensor, log_det_jacobian: torch.Tensor) -> torch.Tensor:
    log_prob = -0.5 * (z**2).sum(dim=1)
    return -(log_prob + log_det_jacobian).mean()


class NormalizingFlowModel(GenAIModelBase):
    torch_module_class = RealNVP

    def __init__(self, torch_module: RealNVP, trainer, sampler, dataset):
        super().__init__(torch_module, trainer, sampler, dataset)

    def train(self) -> None:
        self.trainer.criterion = normalizing_flow_loss
        pass

    def sample(self, save_dir: str, num_samples: int) -> None:
        pass

    def load(self, file_path: str) -> None:
        pass

    def save(self, save_dir: str) -> None:
        """save trained model to save dir.

        Args:
            save_dir (str): save directory.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.torch_module.state_dict(), os.path.join(save_dir, "RealNVP.pth"))
