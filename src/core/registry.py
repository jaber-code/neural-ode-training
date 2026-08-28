"""Generic name -> class registry.

This is the stand-in for "interface + look it up by name": each axis you
want to swap (integrator, dataset, model) gets one Registry here. An
implementation registers itself with a decorator; a config then just names
it, and `.build()` does the lookup + construction. To add a new option you
write a class and add one decorator line -- nothing else needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Generic, Type, TypeVar

# These are only used in type annotations below, never at runtime -- a real
# (non-guarded) import here would be circular: core.integrators.base is only
# reachable by first running core/integrators/__init__.py, which imports
# euler.py, which imports INTEGRATORS from this very module. TYPE_CHECKING
# is always False when Python actually runs this file, so this block never
# executes and the cycle never happens; `from __future__ import annotations`
# (PEP 563) makes every annotation below a lazy string, so nothing at
# runtime ever needs these names to actually exist.
if TYPE_CHECKING:
    from torch import nn

    from core.datasets.base import TransitionDataset
    from core.integrators.base import Integrator
    from core.trainers.base import Trainer

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic over T = the interface every entry implements (Integrator,
    TransitionDataset, ...), so build() can be typed as returning at least
    that interface -- letting IDE features like go-to-definition resolve
    correctly on the built object, even though the exact class is only
    known at runtime from a config string."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Type[T]] = {}

    def register(self, name: str):
        def deco(cls: Type[T]) -> Type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered to {self._items[name]!r}")
            self._items[name] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs: Any) -> T:
        print(f"Building {self.kind} '{name}' with args: {kwargs}")
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. available: {self.names()}")
        return self._items[name](**kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


INTEGRATORS: Registry[Integrator] = Registry("integrator")
DATASETS: Registry[TransitionDataset] = Registry("dataset")
MODELS: Registry[nn.Module] = Registry("model")
TRAINERS: Registry[Trainer] = Registry("trainer")
