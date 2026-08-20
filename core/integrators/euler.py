from torch import Tensor

from core.registry import INTEGRATORS

from .base import Integrator


@INTEGRATORS.register("euler")
class EulerIntegrator(Integrator):
    """Explicit Euler: s_next = s + h * v(s, a). 1 model call per sub-step."""

    def step(self, model, s: Tensor, a: Tensor, h: Tensor) -> Tensor:
        v = model(s, a)
        return s + v * h
