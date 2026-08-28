"""Importing this package registers every dataset implementation below."""

from .gridball import GridballDataset
from .mujoco import MuJoCoDataset

__all__ = ["GridballDataset", "MuJoCoDataset"]
