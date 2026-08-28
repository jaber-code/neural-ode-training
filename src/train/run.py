"""Generic entrypoint: one config file = one experiment.

Usage (from anywhere):
    python train/run.py configs/step1_gridball_euler.yaml

Swap integrator/dataset/model by pointing at a different config, or editing
one `name:` field in this one -- no code changes needed. To add a new
integrator/dataset/model, write a class in core/{integrators,datasets,models}
and register it with a decorator (see any existing file there for the
pattern); it becomes available under its registered name immediately.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import numpy as np
import torch

import core.datasets  # noqa: F401  (import registers implementations)
import core.integrators  # noqa: F401
import core.models  # noqa: F401
import core.trainers  # noqa: F401
from core.config import load_config
from core.engine import evaluate_analytic_sweep, evaluate_convergence, split_dataset
from core.registry import DATASETS, INTEGRATORS, MODELS, TRAINERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a YAML experiment config")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = DATASETS.build(cfg.dataset.name, **cfg.dataset.params)
    model = MODELS.build(
        cfg.model.name, state_dim=dataset.state_dim, action_dim=dataset.action_dim, **cfg.model.params
    )
    integrator = INTEGRATORS.build(cfg.integrator.name, **cfg.integrator.params)
    trainer = TRAINERS.build(cfg.trainer.name, **cfg.trainer.params)

    prepared = trainer.prepare_dataset(dataset)  # identity for single_step; windows the trajectory for multi_step
    train_ds, val_ds = split_dataset(prepared, cfg.train.val_frac, cfg.seed)

    print(
        f"[{cfg.name}]  dataset={cfg.dataset.name} (n={len(dataset)}, "
        f"state_dim={dataset.state_dim}, action_dim={dataset.action_dim})  "
        f"model={cfg.model.name}  integrator={cfg.integrator.name}  trainer={cfg.trainer.name}  "
        f"train/val={len(train_ds)}/{len(val_ds)}"
    )

    trainer.train_loop(model, integrator, train_ds, val_ds, cfg.train, device=cfg.device)

    if cfg.eval.analytic_test_dts:
        evaluate_analytic_sweep(
            model, integrator, dataset, cfg.train.n_sub, cfg.eval.analytic_test_dts,
            device=cfg.device, max_samples=cfg.eval.max_samples,
        )
    if cfg.eval.convergence_dt is not None and cfg.eval.convergence_n_subs:
        evaluate_convergence(
            model, integrator, dataset, cfg.eval.convergence_dt, cfg.eval.convergence_n_subs,
            device=cfg.device, max_samples=cfg.eval.max_samples,
        )

    Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.checkpoint)
    print(f"\nsaved {cfg.checkpoint}")


if __name__ == "__main__":
    main()
