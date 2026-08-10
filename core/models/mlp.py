"""v_theta(s, a) -> ds/dt, a plain MLP. Registered so architecture is also a
config choice (`model.name: mlp`, `model.params.hidden: [128, 128]`) -- swap
in a different network later by adding a class + decorator, same as
integrators/datasets.
"""

from typing import Sequence

import torch
import torch.nn as nn

from core.registry import MODELS


@MODELS.register("mlp")
class VectorFieldMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        dims = [state_dim + action_dim, *hidden, state_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))
