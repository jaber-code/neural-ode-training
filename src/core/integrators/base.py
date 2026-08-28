"""Integrator interface.

Every ODE-integration scheme plugs in here by implementing exactly one
method: how to advance the state by one sub-step of size h, holding the
action a fixed (zero-order hold) and querying the learned field via
model(s, a) -> ds/dt. The shooting rollout used by the training loss
(`integrate`, chopping dt into n_sub sub-steps) is shared by all of them,
so Euler/RK4/etc. differ only in `step`.
"""

from abc import ABC, abstractmethod
from random import random

from torch import Tensor
from typing import List
import torch

class Integrator(ABC):
    @abstractmethod
    def step(self, model, s: Tensor, a: Tensor, h: Tensor) -> Tensor:
        """Advance state s by one sub-step of size h under model(s, a). h and s are
        tensors of matching batch dimension; h is (B, 1), s is (B, state_dim)."""

    def integrate(self, model, s0: Tensor, a: Tensor, dt: Tensor, n_sub: int) -> Tensor:
        """Shooting rollout: split dt into n_sub sub-steps, chain `step` that many
        times, return the endpoint. dt is (B, 1); s0/a are (B, state_dim)/(B, action_dim)."""
        h = dt / n_sub
        s = s0
        for _ in range(n_sub):
            s = self.step(model, s, a, h)
        return s

    def rollout(self, model, s0: Tensor, a: Tensor, dt: Tensor, n_sub: int, perturb_action: bool = False) -> List[Tensor]:
        """Same computation as integrate(), but keeps every intermediate sub-step
        state instead of throwing all but the last one away -- for looking at what
        the model predicts BETWEEN two real recorded points, not just at them.
        Never used during training (that only needs the endpoint); this is purely
        for visualization/inspection. Returns n_sub+1 states:
        [s0, after sub-step 1, after sub-step 2, ..., after sub-step n_sub]
        (that last one is identical to what integrate() alone would return)."""
        h = dt / n_sub
        s = s0
        states = [s]
        original_a = a
        for _ in range(n_sub):
            s = self.step(model, s, a, h)
            if perturb_action:
                a = self._peturb_action(original_a)
            states.append(s)
        return states

    def _peturb_action(self, a: Tensor) -> Tensor:
        #random_peturbations = torch.rand(a.size()) - 0.5
        x = 0.14
        random_peturbations = torch.tensor([x, 0.0, 0.0])
        print("random_peturbations: ", random_peturbations)
        a = a + random_peturbations
        return a
