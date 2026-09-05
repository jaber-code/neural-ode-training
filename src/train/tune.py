"""Hyperparameter search over ONE (dataset, model, integrator) combo, using
Optuna. Separate from train/run.py on purpose: a search trial is a short,
cheap training run whose only job is to rank hyperparameters against each
other, not to produce a real checkpoint, GIFs, or eval sweeps -- none of
run.py's output-producing steps (checkpoint save, evaluate_analytic_sweep,
evaluate_convergence) run here at all.

One study per config, not one global study across every integrator/model/
dataset combo -- see the chat history for why: different integrators have
different per-step cost (RK4 is 4x Euler's model calls), which would bias a
shared search toward whichever is cheaper to try more of, and this project's
actual point is comparing integrators fairly, which needs each one tuned to
its own best rather than whatever a shared search happened to land on.

Dataset/model/integrator/trainer *names* are never touched -- only
train.lr/weight_decay/scheduler(+params) are searched, via the exact same
ComponentConfig.build() calls run.py uses. Needs optuna: pip install optuna.

Usage:
    python train/tune.py configs/step2_mujoco_rk4_multistep.yaml --n_trials 30
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import numpy as np
import torch
import yaml

import core.datasets  # noqa: F401
import core.integrators  # noqa: F401
import core.models  # noqa: F401
import core.trainers  # noqa: F401
from core.config import load_config
from core.engine import split_dataset
from core.registry import DATASETS, INTEGRATORS, MODELS, TRAINERS

try:
    import optuna
except ImportError as e:
    raise ImportError("hyperparameter search needs optuna: pip install optuna") from e


def suggest_model_params(trial: "optuna.Trial", base_cfg) -> dict:
    """Architecture search, conditional on model.name -- mlp and cnn have
    genuinely different knobs, so this is an if/elif on the one thing that
    actually differs, same spirit as run.py's `hasattr(dataset, "frame_shape")`
    branch. Key names here must match model_params_from_best below exactly,
    since that's how the winning trial's flat best_params dict gets turned
    back into a real hidden/channels list afterward."""
    if base_cfg.model.name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden = [trial.suggest_int(f"hidden_{i}", 32, 512, log=True) for i in range(n_layers)]
        return {**base_cfg.model.params, "hidden": hidden}

    if base_cfg.model.name == "cnn":
        n_layers = trial.suggest_int("cnn_n_layers", 2, 4)
        channels = [trial.suggest_int(f"channels_{i}", 16, 128, log=True) for i in range(n_layers)]
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        embed_dim = trial.suggest_int("embed_dim", 8, 64, log=True)
        return {**base_cfg.model.params, "channels": channels, "kernel_size": kernel_size, "embed_dim": embed_dim}

    return dict(base_cfg.model.params)  # unrecognized model name -- leave architecture untouched


def model_params_from_best(best_params: dict, base_cfg) -> dict:
    """Mirror of suggest_model_params, reading already-decided values out of
    study.best_params instead of asking a trial to pick new ones -- used once,
    at the end, to write the winning architecture into the output config."""
    if base_cfg.model.name == "mlp":
        n_layers = best_params["n_layers"]
        hidden = [best_params[f"hidden_{i}"] for i in range(n_layers)]
        return {**base_cfg.model.params, "hidden": hidden}

    if base_cfg.model.name == "cnn":
        n_layers = best_params["cnn_n_layers"]
        channels = [best_params[f"channels_{i}"] for i in range(n_layers)]
        return {
            **base_cfg.model.params,
            "channels": channels,
            "kernel_size": best_params["kernel_size"],
            "embed_dim": best_params["embed_dim"],
        }

    return dict(base_cfg.model.params)


def make_objective(base_cfg, dataset, train_ds, val_ds, search_epochs: int, device: str):
    """Closure over the stuff that's the SAME across every trial (dataset,
    split, base config) -- rebuilding those per trial would be wasted I/O
    (dataset loading, in atari_pong's case a real download check) for
    something no trial's hyperparameters actually change."""

    def objective(trial: "optuna.Trial") -> float:
        train_cfg = copy.deepcopy(base_cfg.train)
        train_cfg.epochs = search_epochs
        train_cfg.distributed = False  # one trial = one process; DDP has nothing to parallelize here

        train_cfg.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        train_cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        train_cfg.scheduler = trial.suggest_categorical("scheduler", ["none", "step", "cosine"])
        if train_cfg.scheduler == "step":
            train_cfg.step_scheduler_params = {
                "step_size": trial.suggest_int("step_size", 5, max(5, search_epochs)),
                "gamma": trial.suggest_float("gamma", 0.1, 0.9),
            }
        elif train_cfg.scheduler == "cosine":
            train_cfg.cosine_scheduler_params = {
                "T_max": trial.suggest_int("T_max", 1, max(1, search_epochs)),
                "eta_min": trial.suggest_float("eta_min", 0.0, 1e-4),
            }

        # same seed every trial -- controls for weight-init randomness, so a
        # difference in val loss reflects the hyperparameters, not lucky init
        torch.manual_seed(base_cfg.seed)
        np.random.seed(base_cfg.seed)

        model_params = suggest_model_params(trial, base_cfg)
        model_kwargs = dict(state_dim=dataset.state_dim, action_dim=dataset.action_dim, **model_params)
        if hasattr(dataset, "frame_shape"):
            model_kwargs["frame_shape"] = dataset.frame_shape
        model = MODELS.build(base_cfg.model.name, **model_kwargs)
        integrator = INTEGRATORS.build(base_cfg.integrator.name, **base_cfg.integrator.params)
        trainer = TRAINERS.build(base_cfg.trainer.name, **base_cfg.trainer.params)

        history = trainer.train_loop(model, integrator, train_ds, val_ds, train_cfg, device=device)
        final_val_mse = history[-1][2]  # (epoch, train_mse, val_mse) tuples; final epoch's val loss
        return final_val_mse

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a YAML experiment config -- fixes dataset/model/integrator")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--search_epochs", type=int, default=5, help="epochs per trial -- short on purpose, just to rank")
    parser.add_argument("--storage", default=None, help="optuna storage URL, e.g. sqlite:///output/optuna.db -- omit for in-memory (no file written)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"tuning [{cfg.name}]  dataset={cfg.dataset.name}  model={cfg.model.name}  integrator={cfg.integrator.name}  "
          f"trainer={cfg.trainer.name}  ({args.n_trials} trials x {args.search_epochs} epochs)")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dataset = DATASETS.build(cfg.dataset.name, **cfg.dataset.params)
    trainer_for_split = TRAINERS.build(cfg.trainer.name, **cfg.trainer.params)
    prepared = trainer_for_split.prepare_dataset(dataset)
    train_ds, val_ds = split_dataset(prepared, cfg.train.val_frac, cfg.seed)

    study = optuna.create_study(
        study_name=cfg.name, direction="minimize",
        storage=args.storage, load_if_exists=args.storage is not None,
    )
    study.optimize(make_objective(cfg, dataset, train_ds, val_ds, args.search_epochs, device), n_trials=args.n_trials)

    print(f"\nbest trial: val_mse={study.best_value:.6f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # winning hyperparameters, folded back into a normal runnable config --
    # the actual deliverable, since re-typing best_params by hand into a
    # config invites transcription mistakes
    tuned_cfg = copy.deepcopy(cfg)
    tuned_cfg.train.lr = study.best_params["lr"]
    tuned_cfg.train.weight_decay = study.best_params["weight_decay"]
    tuned_cfg.train.scheduler = study.best_params["scheduler"]
    if tuned_cfg.train.scheduler == "step":
        tuned_cfg.train.step_scheduler_params = {
            "step_size": study.best_params["step_size"], "gamma": study.best_params["gamma"],
        }
    elif tuned_cfg.train.scheduler == "cosine":
        tuned_cfg.train.cosine_scheduler_params = {
            "T_max": study.best_params["T_max"], "eta_min": study.best_params["eta_min"],
        }
    tuned_cfg.model.params = model_params_from_best(study.best_params, cfg)

    out_path = Path(args.config).with_name(f"{Path(args.config).stem}_tuned.yaml")
    with open(out_path, "w") as f:
        yaml.dump(_config_to_dict(tuned_cfg), f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {out_path} -- a full config with these hyperparameters, ready for a real (full-length) train/run.py run")


def _config_to_dict(cfg) -> dict:
    import dataclasses
    return dataclasses.asdict(cfg)


if __name__ == "__main__":
    main()
