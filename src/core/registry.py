"""Generic name -> class registry.

This is the stand-in for "interface + look it up by name": each axis you
want to swap (integrator, dataset, model) gets one Registry here. An
implementation registers itself with a decorator; a config then just names
it, and `.build()` does the lookup + construction. To add a new option you
write a class and add one decorator line -- nothing else needs to change.
"""

from __future__ import annotations

import inspect
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
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. available: {self.names()}")
        cls = self._items[name]

        # Tolerant construction: a config carrying a param left over from a different
        # dataset/model/trainer/integrator (e.g. copy-pasted between configs) shouldn't
        # crash the whole run -- drop what `cls`'s constructor doesn't accept and warn,
        # instead of a bare TypeError. inspect.signature(cls) (the class, not cls.__init__)
        # is what correctly reports "()" for a class with no __init__ of its own -- going
        # through cls.__init__ directly reports the misleading inherited "(*args, **kwargs)"
        # from object.__init__, which would silently let stray kwargs through here only for
        # them to blow up later at the real object.__init__ call.
        params = inspect.signature(cls).parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_var_kwargs:
            used_kwargs, dropped = kwargs, {}
        else:
            used_kwargs = {k: v for k, v in kwargs.items() if k in params}
            dropped = {k: v for k, v in kwargs.items() if k not in params}

        if dropped:
            print(f"WARNING: {self.kind} '{name}' ({cls.__name__}) doesn't accept {sorted(dropped)} -- "
                  f"ignoring (check your config for stray params copied from a different {self.kind})")

        print(f"Building {self.kind} '{name}' with args: {used_kwargs}")
        return cls(**used_kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


INTEGRATORS: Registry[Integrator] = Registry("integrator")
DATASETS: Registry[TransitionDataset] = Registry("dataset")
MODELS: Registry[nn.Module] = Registry("model")
TRAINERS: Registry[Trainer] = Registry("trainer")
