"""Importing this package registers every dataset implementation below."""

from .atari_pong import AtariPongDataset
from .gridball import GridballDataset
from .mujoco import MuJoCoDataset

__all__ = ["AtariPongDataset", "GridballDataset", "MuJoCoDataset"]
