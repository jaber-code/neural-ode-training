"""Importing this package registers every model implementation below."""

from .cnn import VectorFieldCNN
from .mlp import VectorFieldMLP

__all__ = ["VectorFieldCNN", "VectorFieldMLP"]
