"""Offline MuJoCo robotics benchmarks (D4RL-style dumps, e.g. Hopper/
HalfCheetah/Walker2d medium/expert). Unlike the toy generator, real logged
transitions don't carry a per-row dt -- the whole file was recorded at one
fixed control rate -- so `dt` is a single config constant broadcast to every
row.

Expects a .npz or .hdf5 file exposing D4RL-style arrays:
    observations       (N, state_dim)
    actions            (N, action_dim)
    next_observations  (N, state_dim)   -- if absent, derived as a shifted
                                            view of `observations` (see below)
    terminals / timeouts (N,)           -- optional; rows at an episode
                                            boundary are dropped since s2
                                            there is not "one dt later"

No ground-truth dynamics are known for real data, so `analytic_step` stays
the base-class default (None) and the dt-sweep eval is skipped automatically.
"""

from pathlib import Path
from typing import cast

import numpy as np
import torch

from core.registry import DATASETS

from .base import TransitionDataset

@DATASETS.register("mujoco")
class MuJoCoDataset(TransitionDataset):
    supports_windows = True  # rows are one recorded trajectory in order (episode-boundary rows already dropped)

    def __init__(
        self,
        path: str,
        dt: float,
        obs_key: str = "observations",
        action_key: str = "actions",
        next_obs_key: str = "next_observations",
        terminal_key: str = "terminals",
        timeout_key: str = "timeouts",
        normalize: bool = False,
    ):
        keys = [obs_key, action_key, next_obs_key, terminal_key, timeout_key]
        arrays = self._load_arrays(Path(path), keys)
        obs = np.asarray(arrays[obs_key], dtype=np.float32)
        act = np.asarray(arrays[action_key], dtype=np.float32)

        if next_obs_key in arrays:
            next_obs = np.asarray(arrays[next_obs_key], dtype=np.float32)
        else:
            # no explicit next-state column: pair consecutive rows within the
            # same trajectory (drops the last row, which has no successor)
            next_obs, obs, act = obs[1:], obs[:-1], act[:-1]

        valid = np.ones(len(obs), dtype=bool)
        for key in (terminal_key, timeout_key):
            if key in arrays:
                done = np.asarray(arrays[key]).astype(bool)[: len(valid)]
                valid &= ~done  # drop transitions that cross an episode boundary

        obs, act, next_obs = obs[valid], act[valid], next_obs[valid]

        self.state_dim = obs.shape[1]
        self.action_dim = act.shape[1]
        self.dt = float(dt)

        if normalize:
            self.obs_mean = obs.mean(axis=0)
            self.obs_std = obs.std(axis=0) + 1e-6
            obs = (obs - self.obs_mean) / self.obs_std
            next_obs = (next_obs - self.obs_mean) / self.obs_std

        self.s1 = torch.from_numpy(obs)
        self.a = torch.from_numpy(act)
        self.s2 = torch.from_numpy(next_obs)

    @staticmethod
    def _load_arrays(path: Path, keys: list[str]) -> dict:
        """Fetch only the named top-level datasets. D4RL-style files also
        contain nested groups (infos/qpos, metadata/policy/...) that aren't
        plain arrays and aren't needed here, so we look up each key
        directly instead of iterating every entry in the file."""
        if path.suffix == ".npz":
            with np.load(path) as npz:
                return {k: npz[k] for k in keys if k in npz}
        if path.suffix in (".hdf5", ".h5"):
            try:
                import h5py
            except ImportError as e:
                raise ImportError(
                    "reading a .hdf5 MuJoCo dataset needs h5py: pip install h5py"
                ) from e
            with h5py.File(path, "r") as f:
                # f[k] is typed as Dataset | Group | Datatype; every key we
                # ask for is a plain array in these files, never a group or
                # a named datatype, so this narrows it back to Dataset.
                return {k: cast(h5py.Dataset, f[k])[:] for k in keys if k in f}
        raise ValueError(f"unsupported MuJoCo dataset file type: {path.suffix}")

    def __len__(self) -> int:
        return self.s1.shape[0]

    def __getitem__(self, idx):
        dt = torch.tensor([self.dt], dtype=torch.float32)
        return self.s1[idx], self.a[idx], dt, self.s2[idx]
