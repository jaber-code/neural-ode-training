"""Experiment config: one YAML file = one experiment. Each of dataset/model/
integrator is just a `name` (registry key) + free-form `params` dict that
gets passed straight to that class's constructor -- so adding a new field
to e.g. GridballDataset.__init__ never requires touching this file.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ComponentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentConfig":
        return cls(name=d["name"], params=d.get("params", {}) or {})


@dataclass
class TrainConfig:
    n_sub: int = 8          # integration sub-steps per shooting rollout during training
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    val_frac: float = 0.1
    scheduler: str = "none"                             # "none" | "step" | "cosine" -- picks which params dict below is used
    step_scheduler_params: dict[str, Any] = field(default_factory=lambda: {"step_size": 10, "gamma": 0.5})
    cosine_scheduler_params: dict[str, Any] = field(default_factory=lambda: {"T_max": 30, "eta_min": 0.0})
    distributed: bool = False   # true = multi-GPU DDP; needs `torchrun --nproc_per_node=N`, see core/distributed.py


@dataclass
class EvalConfig:
    analytic_test_dts: list[float] = field(default_factory=list)   # skipped if dataset has no analytic_step
    convergence_dt: Optional[float] = None                          # skipped if None
    convergence_n_subs: list[int] = field(default_factory=list)
    max_samples: Optional[int] = 5000                               # cap eval batch size on large datasets
    perturb_action: bool = False   # master switch: False always forces "none" regardless of
                                    # core.integrators.base.PERTURBATION; True lets that string pick one of the 4



@dataclass
class ExperimentConfig:
    name: str
    dataset: ComponentConfig
    model: ComponentConfig
    integrator: ComponentConfig
    seed: int = 0
    device: str = "cpu"
    trainer: ComponentConfig = field(default_factory=lambda: ComponentConfig("single_step"))
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: str = "output/model.pt"


def load_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        name=raw["name"],
        dataset=ComponentConfig.from_dict(raw["dataset"]),
        model=ComponentConfig.from_dict(raw["model"]),
        integrator=ComponentConfig.from_dict(raw["integrator"]),
        seed=raw.get("seed", 0),
        device=raw.get("device", "cpu"),
        trainer=ComponentConfig.from_dict(raw["trainer"]) if "trainer" in raw else ComponentConfig("single_step"),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        checkpoint=raw.get("checkpoint", "output/model.pt"),
    )
