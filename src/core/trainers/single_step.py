import torch.nn.functional as F

from core.registry import TRAINERS

from .base import Trainer


@TRAINERS.register("single_step")
class SingleStepTrainer(Trainer):
    """The original shooting loss: integrate one dt from a real (s1, a) and
    match the one real endpoint s2. Never sees its own prediction error as
    input, so it has no pressure to be robust to closed-loop drift."""

    def compute_loss(self, model, integrator, batch, n_sub: int):
        s1, a, dt, s2 = batch
        pred = integrator.integrate(model, s1, a, dt, n_sub)
        return F.mse_loss(pred, s2)
