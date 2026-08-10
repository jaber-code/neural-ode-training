"""The toy velocity-grid generator's format: a flat table of rows
    [s1 (state_dim), a (action_dim), dt (1), s2 (state_dim)]
saved by gen/gen_fixed_dt.py / gen_mul_dt.py, as either .npy or .csv (the
generators write both). Point `path` at either one -- same class.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from core.registry import DATASETS

from .base import TransitionDataset


@DATASETS.register("gridball")
class GridballDataset(TransitionDataset):
    def __init__(
        self,
        path: str,
        state_dim: int = 2,
        action_dim: int = 2,
        lo: Optional[Sequence[float]] = None,
        hi: Optional[Sequence[float]] = None,
    ):
        """lo/hi (optional): box bounds for the known clip(s1 + a*dt, lo, hi)
        dynamics used by the gridball toy. Set them to enable the analytic
        dt-sweep eval in core.engine; leave unset if ground truth isn't known."""
        p = Path(path)
        arr = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, delimiter=",", skiprows=1)
        self.data = torch.as_tensor(arr, dtype=torch.float32)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self._lo = lo
        self._hi = hi

        i = 0
        self._s1 = slice(i, i + state_dim); i += state_dim
        self._a = slice(i, i + action_dim); i += action_dim
        self._dt = slice(i, i + 1); i += 1
        self._s2 = slice(i, i + state_dim)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[self._s1], row[self._a], row[self._dt], row[self._s2]

    def analytic_step(self, s1, a, dt):
        if self._lo is None or self._hi is None:
            return None
        lo = torch.as_tensor(self._lo, dtype=s1.dtype, device=s1.device)
        hi = torch.as_tensor(self._hi, dtype=s1.dtype, device=s1.device)
        return torch.clamp(s1 + a * dt, lo, hi)
