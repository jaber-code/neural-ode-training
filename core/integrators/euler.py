from core.registry import INTEGRATORS

from .base import Integrator


@INTEGRATORS.register("euler")
class EulerIntegrator(Integrator):
    """Explicit Euler: s_next = s + h * v(s, a). 1 model call per sub-step."""

    def step(self, model, s, a, h):
        v = model(s, a)
        return s + v * h
