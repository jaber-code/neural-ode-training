"""Dataset-agnostic, integrator-agnostic training + evaluation. This is the
one copy of the loop that used to be duplicated (with small drifted
differences) across train.py / train_one_step.py / train_shooting_vardt.py.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from core.config import TrainConfig
from core.datasets.base import TransitionDataset
from core.integrators.base import Integrator


def split_dataset(dataset: TransitionDataset, val_frac: float, seed: int):
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=g)


def shooting_loss(model, integrator: Integrator, s1, a, dt, s2, n_sub: int):
    pred = integrator.integrate(model, s1, a, dt, n_sub)
    return F.mse_loss(pred, s2)


def train_loop(model, integrator: Integrator, train_ds, val_ds, cfg: TrainConfig, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(len(val_ds), 1), shuffle=False)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for s1, a, dt, s2 in train_loader:
            s1, a, dt, s2 = s1.to(device), a.to(device), dt.to(device), s2.to(device)
            loss = shooting_loss(model, integrator, s1, a, dt, s2, cfg.n_sub)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * s1.shape[0]
            seen += s1.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss, val_seen = 0.0, 0
            for s1, a, dt, s2 in val_loader:
                s1, a, dt, s2 = s1.to(device), a.to(device), dt.to(device), s2.to(device)
                val_loss += shooting_loss(model, integrator, s1, a, dt, s2, cfg.n_sub).item() * s1.shape[0]
                val_seen += s1.shape[0]

        train_mse, val_mse = running / max(seen, 1), val_loss / max(val_seen, 1)
        history.append((epoch, train_mse, val_mse))
        print(f"epoch {epoch:3d}   train endpoint-MSE = {train_mse:.6f}   val endpoint-MSE = {val_mse:.6f}")
    return history


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
