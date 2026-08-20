from torch import Tensor

from core.registry import INTEGRATORS

from .base import Integrator


@INTEGRATORS.register("rk4")
class RK4Integrator(Integrator):
    """Classic 4th-order Runge-Kutta. 4 model calls per sub-step, most accurate
    per sub-step of the three, at 4x the cost of Euler."""

    def step(self, model, s: Tensor, a: Tensor, h: Tensor) -> Tensor:
        k1 = model(s, a)
        k2 = model(s + 0.5 * h * k1, a)
        k3 = model(s + 0.5 * h * k2, a)
        k4 = model(s + h * k3, a)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
