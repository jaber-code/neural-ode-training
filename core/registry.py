"""Generic name -> class registry.

This is the stand-in for "interface + look it up by name": each axis you
want to swap (integrator, dataset, model) gets one Registry here. An
implementation registers itself with a decorator; a config then just names
it, and `.build()` does the lookup + construction. To add a new option you
write a class and add one decorator line -- nothing else needs to change.
"""

from typing import Any, Dict, Type


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Type] = {}

    def register(self, name: str):
        def deco(cls: Type) -> Type:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered to {self._items[name]!r}")
            self._items[name] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs: Any):
        print(f"Building {self.kind} '{name}' with args: {kwargs}")
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. available: {self.names()}")
        return self._items[name](**kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


INTEGRATORS = Registry("integrator")
DATASETS = Registry("dataset")
MODELS = Registry("model")
