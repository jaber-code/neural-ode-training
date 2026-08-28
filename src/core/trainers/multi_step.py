"""Multi-step (rollout) training: unroll K real consecutive hops, feeding the
model's OWN predicted state back in as the input to the next hop (only the
action is ever the real recorded one), and penalize deviation from the real
trajectory at every one of the K points, not just the last. This is what
exposes the model to its own compounding error during training, instead of
only ever seeing perfect ground-truth inputs -- the fix for the closed-loop
divergence seen in plotters/render_mujoco_rollout.py.
"""

from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from core.datasets.base import TransitionDataset
from core.registry import TRAINERS

from .base import Trainer


class _OrderedTrajectory(Protocol):
    """The shape MultiStepWindowDataset actually needs: ordered per-row
    tensors and a fixed dt. TransitionDataset itself doesn't declare these
    (GridballDataset has no such attributes), so this documents the extra
    contract that `supports_windows=True` implies, letting the type checker
    verify usage below instead of treating every access as unknown."""

    s1: Tensor
    a: Tensor
    s2: Tensor
    dt: float

    def __len__(self) -> int: ...


class MultiStepWindowDataset(Dataset):
    """Wraps a TransitionDataset with supports_windows=True (rows form one
    ordered trajectory, not i.i.d. samples) and yields valid length-K windows
    of consecutive real transitions: a real starting state, the K real
    actions applied from there, and the K real states that followed.

    Not registered in DATASETS -- this is a view over an existing dataset,
    not a data source of its own; built internally by
    MultiStepTrainer.prepare_dataset.
    """

    def __init__(self, base: TransitionDataset, k: int, atol: float = 1e-5):
        if not getattr(base, "supports_windows", False):
            raise ValueError(
                f"{type(base).__name__} doesn't support multi-step windows -- its rows "
                f"are independent samples, not an ordered trajectory (e.g. gridball draws "
                f"a fresh random action and dt per row, so there's no real 'next transition' "
                f"to chain). Multi-step training needs a trajectory-shaped source, e.g. mujoco."
            )
        if k < 1:
            raise ValueError(f"window length k must be >= 1, got {k}")
        # supports_windows=True is exactly the runtime guarantee that base has
        # s1/a/s2/dt -- verified above, so this cast just gives the type
        # checker the same information.
        self.base = cast(_OrderedTrajectory, base)
        self.k = k
        self.starts = self._valid_starts(atol)

    def _valid_starts(self, atol: float) -> torch.Tensor:
        n, k = len(self.base), self.k
        if n < k:
            return torch.empty(0, dtype=torch.long)
        # adjacent[j] True iff row j+1 genuinely follows row j (base.s1[j+1] == base.s2[j])
        adjacent = (self.base.s1[1:] - self.base.s2[:-1]).abs().amax(dim=1) < atol  # (n-1,)
        prefix = torch.cat([torch.zeros(1, dtype=torch.long), adjacent.long().cumsum(0)])  # (n,)
        # a window starting at i needs k-1 consecutive True values: adjacent[i .. i+k-2]
        n_starts = n - k + 1
        run_len = prefix[k - 1 : k - 1 + n_starts] - prefix[0:n_starts]
        return torch.nonzero(run_len == k - 1, as_tuple=True)[0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        s0 = self.base.s1[i]
        actions = self.base.a[i : i + self.k]        # (k, action_dim)
        true_states = self.base.s2[i : i + self.k]   # (k, state_dim)
        dt = torch.tensor([self.base.dt], dtype=torch.float32)
        return s0, actions, dt, true_states


@TRAINERS.register("multi_step")
class MultiStepTrainer(Trainer):
    def __init__(self, k: int):
        self.k = k

    def prepare_dataset(self, dataset: TransitionDataset) -> Dataset:
        return MultiStepWindowDataset(dataset, self.k)

    def compute_loss(self, model, integrator, batch, n_sub: int):
        s0, actions, dt, true_states = batch  # (B,state_dim) (B,k,action_dim) (B,1) (B,k,state_dim)
        s = s0
        losses = []
        for t in range(actions.shape[1]):
            s = integrator.integrate(model, s, actions[:, t], dt, n_sub)
            losses.append(F.mse_loss(s, true_states[:, t]))
        return torch.stack(losses).mean()
