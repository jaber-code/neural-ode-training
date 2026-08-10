"""Integrator interface.

Every ODE-integration scheme plugs in here by implementing exactly one
method: how to advance the state by one sub-step of size h, holding the
action a fixed (zero-order hold) and querying the learned field via
model(s, a) -> ds/dt. The shooting rollout used by the training loss
(`integrate`, chopping dt into n_sub sub-steps) is shared by all of them,
so Euler/RK4/etc. differ only in `step`.
"""

from abc import ABC, abstractmethod


class Integrator(ABC):
    @abstractmethod
    def step(self, model, s, a, h):
        """Advance state s by one sub-step of size h under model(s, a). h and s are
        tensors of matching batch dimension; h is (B, 1), s is (B, state_dim)."""

    def integrate(self, model, s0, a, dt, n_sub: int):
        """Shooting rollout: split dt into n_sub sub-steps, chain `step` that many
        times, return the endpoint. dt is (B, 1); s0/a are (B, state_dim)/(B, action_dim)."""
        h = dt / n_sub
        s = s0
        for _ in range(n_sub):
            s = self.step(model, s, a, h)
        return s
