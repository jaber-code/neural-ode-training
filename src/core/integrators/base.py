"""Integrator interface.

Every ODE-integration scheme plugs in here by implementing exactly one
method: how to advance the state by one sub-step of size h, holding the
action a fixed (zero-order hold) and querying the learned field via
model(s, a) -> ds/dt. The shooting rollout used by the training loss
(`integrate`, chopping dt into n_sub sub-steps) is shared by all of them,
so Euler/RK4/etc. differ only in `step`.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

# Hardcoded offset for the two "offset_*" perturbation modes below --
# same [-0.2, 0, 0.2] vector used throughout the earlier action-perturbation
# experiments in this file's history.
PERTURBATION_OFFSET = [-0.2, 0.0, 0.2]


def _action_none(original_a: Tensor, i: int, n_sub: int) -> Tensor:
    """1. No change -- always the real recorded action."""
    return original_a


def _action_zero(original_a: Tensor, i: int, n_sub: int) -> Tensor:
    """2. Zero action on every sub-step, from the very first one -- simulates
    cutting all control input entirely (a "power cut" test)."""
    return torch.zeros_like(original_a)


def _action_offset_from_midpoint(original_a: Tensor, i: int, n_sub: int) -> Tensor:
    """3. original_a + PERTURBATION_OFFSET, held constant from the midpoint
    sub-step onward -- a sustained perturbation for the second half of the rollout."""
    if i >= n_sub // 2:
        offset = torch.tensor(PERTURBATION_OFFSET, device=original_a.device, dtype=original_a.dtype)
        return torch.clamp(original_a + offset.expand_as(original_a), -1.0, 1.0)
    return original_a


def _action_offset_at_midpoint(original_a: Tensor, i: int, n_sub: int) -> Tensor:
    """4. original_a + PERTURBATION_OFFSET, applied ONLY on the exact midpoint
    sub-step -- a single-step pulse, reverting to the real action every other step."""
    if i == n_sub // 2:
        offset = torch.tensor(PERTURBATION_OFFSET, device=original_a.device, dtype=original_a.dtype)
        return torch.clamp(original_a + offset.expand_as(original_a), -1.0, 1.0)
    return original_a


PERTURBATIONS = {
    "none": _action_none,
    "zero": _action_zero,
    "offset_from_midpoint": _action_offset_from_midpoint,
    "offset_at_midpoint": _action_offset_at_midpoint,
}

# THE string that decides which of the 4 functions above rollout() uses --
# edit this to switch what every render/inspection run does. Not a rollout()
# parameter on purpose: one shared choice here, read by rollout() below and
# by anything (e.g. render_mujoco_rollout.py) that wants to label output with
# whichever mode actually ran.
PERTURBATION = "offset_from_midpoint"


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

    def rollout(self, model, s0: Tensor, a: Tensor, dt: Tensor, n_sub: int) -> List[Tensor]:
        """Same computation as integrate(), but keeps every intermediate sub-step
        state instead of throwing all but the last one away -- for looking at what
        the model predicts BETWEEN two real recorded points, not just at them.
        Never used during training (that only needs the endpoint); this is purely
        for visualization/inspection. Returns n_sub+1 states:
        [s0, after sub-step 1, after sub-step 2, ..., after sub-step n_sub]
        (that last one is identical to what integrate() alone would return).

        Which action actually gets used at each sub-step is decided by the
        module-level PERTURBATION string above, one of PERTURBATIONS' keys --
        edit that to switch modes, not this method's arguments."""
        apply_perturbation = PERTURBATIONS[PERTURBATION]

        h = dt / n_sub
        s = s0
        states = [s]
        original_a = a
        for i in range(1, n_sub + 1):
            a = apply_perturbation(original_a, i, n_sub)
            s = self.step(model, s, a, h)
            states.append(s)
        return states
