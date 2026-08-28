"""Dataset interface.

Every data source -- a toy generator's .npy dump, a text file, a real
offline-RL benchmark -- implements this once and everything downstream
(training loop, shooting loss, evaluators) works unchanged. The contract:
given an index, return one observed (s1, a, dt, s2) transition, as 1D
float32 tensors. `state_dim`/`action_dim` tell the model how to size its
input/output layers.

`analytic_step` is optional: toy environments with known closed-form
dynamics (e.g. clip(s1 + a*dt)) can override it to enable the dt-sweep
ground-truth check in core.engine. Real datasets have no such ground
truth, so the default (return None) makes that check silently skip.

`supports_windows` is also optional, for multi-step rollout training
(core.trainers.multi_step): it's only true for sources whose rows are a
single recorded trajectory in order, where row i+1 genuinely follows row i
(e.g. mujoco). Datasets of i.i.d. sampled transitions (e.g. gridball, where
every row is an independently random state/action/dt) have no such
adjacency, so the default (False) makes multi-step training refuse them
with a clear error instead of silently training on nonsense windows.
"""

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset


class TransitionDataset(Dataset, ABC):
    state_dim: int
    action_dim: int
    supports_windows: bool = False

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """-> (s1, a, dt, s2)"""

    def analytic_step(self, s1: Tensor, a: Tensor, dt: Tensor) -> Optional[Tensor]:
        return None
