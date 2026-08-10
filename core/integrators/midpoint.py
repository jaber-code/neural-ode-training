from core.registry import INTEGRATORS

from .base import Integrator


@INTEGRATORS.register("midpoint")
class MidpointIntegrator(Integrator):
    """Explicit midpoint (RK2): evaluate the field once at the Euler half-step,
    then take the full step using that midpoint slope. 2 model calls per sub-step."""

    def step(self, model, s, a, h):
        v1 = model(s, a)
        v2 = model(s + 0.5 * h * v1, a)
        return s + h * v2
