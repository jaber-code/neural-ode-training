"""Dataset-agnostic, integrator-agnostic, trainer-agnostic evaluation, plus
the train/val split every trainer shares. The training loop itself
(shared epoch skeleton + swappable loss) lives in core.trainers -- see
core/trainers/base.py for why that split exists.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split

from core.datasets.base import TransitionDataset
from core.integrators.base import Integrator


def split_dataset(dataset, val_frac: float, seed: int):
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


def _load_full_batch(dataset, device: str, max_samples: Optional[int] = None):
    n = len(dataset)
    if max_samples is not None and n > max_samples:
        idx = torch.randperm(n)[:max_samples].tolist()
        dataset = Subset(dataset, idx)
        n = max_samples
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    s1, a, dt, s2 = next(iter(loader))
    return s1.to(device), a.to(device), dt.to(device), s2.to(device)


def evaluate_analytic_sweep(model, integrator: Integrator, dataset: TransitionDataset, n_sub: int,
                             test_dts: list[float], device: str = "cpu", max_samples: Optional[int] = None):
    """Compare model rollout vs known ground-truth dynamics at several dt values,
    including some the training distribution didn't emphasize. Requires the
    dataset to override `analytic_step`; silently no-ops otherwise (e.g. for
    real offline-RL data with no closed-form dynamics)."""
    s1, a, _, _ = _load_full_batch(dataset, device, max_samples)
    print("\n" + "=" * 55)
    model.eval()
    for test_dt in test_dts:
        dt_col = torch.full((s1.shape[0], 1), test_dt, device=device)
        true = dataset.analytic_step(s1, a, dt_col)
        if true is None:
            print("dataset has no analytic ground truth -- skipping dt-sweep eval")
            return
        with torch.no_grad():
            pred = integrator.integrate(model, s1, a, dt_col, n_sub)
        err = (pred - true).norm(dim=1)
        print(f"dt={test_dt:<5}  mean-endpoint-err={err.mean():.4f}  worst={err.max():.4f}")


def evaluate_convergence(model, integrator: Integrator, dataset: TransitionDataset, dt: float,
                          n_subs: list[int], device: str = "cpu", max_samples: Optional[int] = None):
    """Ground-truth-free stability check: does the rollout endpoint stabilize as
    sub-steps increase? Large drift between consecutive n_sub values means the
    learned field is too stiff/discontinuous for coarse integration -- useful
    on real data where no analytic dynamics exist to compare against."""
    s1, a, _, _ = _load_full_batch(dataset, device, max_samples)
    dt_col = torch.full((s1.shape[0], 1), dt, device=device)
    print("\nintegration self-consistency (endpoint drift vs previous n_sub):")
    model.eval()
    prev = None
    for n_sub in n_subs:
        with torch.no_grad():
            pred = integrator.integrate(model, s1, a, dt_col, n_sub)
        drift = (pred - prev).norm(dim=1).mean().item() if prev is not None else float("nan")
        print(f"n_sub={n_sub:<4} drift={drift:.6f}")
        prev = pred
