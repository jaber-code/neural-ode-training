# Learning continuous-time dynamics from logged transitions

Thesis code. The goal: given only recorded transitions `(s1, a, dt, s2)` — a
state, an action, a time gap, and the state that followed — learn a
**continuous-time vector field** `v_θ(s, a) ≈ ds/dt` that also captures the
motion *between* two recorded observations, not just the jump from one to the
next.

## The idea

A plain one-step model learns `s2 = f(s1, a)` directly and says nothing about
what happens inside the interval `dt`. Here the model instead predicts a
velocity, and we get `s2` by **numerically integrating** that velocity over
`dt`:

- split `dt` into `n_sub` small sub-steps,
- advance the state through each one with a classic ODE integrator
  (Euler / midpoint / RK4), holding the action fixed,
- compare the final state to the real `s2` and backprop through the whole
  rollout (a "shooting" loss).

This forces the field to be consistent at any step size, so we can then roll it
out at finer resolution and inspect the in-between trajectory.

## How the code is organised

One **YAML config = one experiment**. Dataset, model, integrator, and trainer
are each just a name plus a `params` dict; a registry looks the name up and
builds the object. Adding a new integrator/dataset/model/trainer is a new class
plus one `@register(...)` line — no changes to the training loop or config
parser.

| Piece | Interface | What varies |
|-------|-----------|-------------|
| [Integrator](src/core/integrators/base.py) | `step(model, s, a, h)` | one sub-step rule (Euler, midpoint, RK4) |
| [Trainer](src/core/trainers/base.py) | `compute_loss(...)`, `prepare_dataset(...)` | single-endpoint vs. multi-step rollout loss |
| [Dataset](src/core/datasets/base.py) | `__getitem__ -> (s1, a, dt, s2)` | toy generator, offline MuJoCo, Atari frames |
| [Model](src/core/models/mlp.py) | `forward(s, a) -> ds/dt` | MLP for vectors, CNN for pixels |

```
src/core/         registry, config, training engine, and the four pluggable pieces
src/train/run.py  entrypoint: python src/train/run.py <config.yaml>
src/plotters/     render a trained model's rollout vs. the real trajectory as a GIF
configs/          one file per experiment
```

## Tech stack & engineering

- **Python**, PyTorch, NumPy, PyYAML — no framework beyond that.
- **Modern architecture patterns**: a name→class registry for dependency
  injection, the template-method pattern on both `Integrator` and `Trainer`
  (shared skeleton, one overridable method), and dataclass-typed configs. New
  behaviour is added by writing a class, not by editing existing code.
- **Distributed training**: multi-GPU `DistributedDataParallel`, launchable
  either via `torchrun` or directly by the SLURM scheduler (rank read from
  `SLURM_PROCID`), toggled by a single config flag.
- **Automation**: config-driven runs (one file fully describes an experiment),
  automatic dataset download from the HuggingFace Hub, checkpoints tagged with
  the SLURM job id so a saved model traces back to its run log, and SLURM +
  container (`enroot`) batch scripts for the cluster.
- All code and docstrings are in English; each module documents *why* it is
  shaped the way it is, not just what it does.

## Running

```bash
python src/train/run.py configs/step1_gridball_euler.yaml
```

Training prints per-epoch train/val loss, runs the evaluation sweeps, and saves
a checkpoint to `output/<name>_<runid>.pt`. Multi-GPU (DDP) and SLURM launch
scripts are included; set `train.distributed: true` to enable it.

## Experiment steps

1. **Gridball** (`configs/step1_*`) — a 2D toy with *known* closed-form
   dynamics. Used to validate the pipeline and compare integrators against
   ground truth across a range of `dt` values.
2. **Offline MuJoCo** (`configs/step2_*`) — real logged Hopper data, no ground
   truth. Evaluated with an integration self-consistency check and rendered
   rollouts. A **multi-step trainer** (unroll `k` real steps, feed the model
   its own predictions back in) was added here to fix closed-loop drift.
3. **Atari Pong** (`configs/step3_*`) — pixel states with a CNN vector field
   and an action embedding. Work in progress.

## Evaluation

- **Analytic dt-sweep** — rollout vs. true dynamics at several step sizes,
  including ones the training data didn't emphasise (toy data only).
- **Integration self-consistency** — does the rollout endpoint stop moving as
  `n_sub` increases? Large drift means the learned field is too stiff for
  coarse integration. Works without any ground truth.
- **Rollout rendering** — replay a real action sequence through the model and
  animate the predicted motion next to the recorded one.
