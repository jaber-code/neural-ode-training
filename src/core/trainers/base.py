"""Trainer interface -- same template-method shape as core.integrators.base.

`train_loop` (the epoch loop: iterate batches, compute loss, backward, clip,
step, log) is shared by every trainer and never needs to change. What
varies between "match one endpoint" and "unroll K real steps and match all
of them" is only the loss computation itself, plus -- unlike Integrator --
what a batch even *is*, since multi-step training needs windows of several
consecutive real transitions instead of single triples. That's the second
override point: `prepare_dataset` lets a trainer wrap the raw dataset into
whatever view its `compute_loss` expects, before the usual train/val split.
"""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset

from core.config import TrainConfig
from core.datasets.base import TransitionDataset
from core.integrators.base import Integrator


class Trainer(ABC):
    def prepare_dataset(self, dataset: TransitionDataset) -> Dataset:
        """Default: use the dataset as-is (one row = one training sample)."""
        return dataset

    @abstractmethod
    def compute_loss(self, model, integrator: Integrator, batch, n_sub: int) -> torch.Tensor:
        """batch is whatever prepare_dataset's output yields per sample, collated
        into a batch by the DataLoader; must return a scalar loss."""

    def train_loop(self, model, integrator: Integrator, train_ds: Dataset, val_ds: Dataset,
                    cfg: TrainConfig, device: str = "cpu"):
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=max(len(val_ds), 1), shuffle=False)

        history = []
        for epoch in range(cfg.epochs):
            model.train()
            running, seen = 0.0, 0
            for batch in train_loader:
                batch = tuple(x.to(device) for x in batch)
                loss = self.compute_loss(model, integrator, batch, cfg.n_sub)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                running += loss.item() * batch[0].shape[0]
                seen += batch[0].shape[0]

            model.eval()
            with torch.no_grad():
                val_loss, val_seen = 0.0, 0
                for batch in val_loader:
                    batch = tuple(x.to(device) for x in batch)
                    val_loss += self.compute_loss(model, integrator, batch, cfg.n_sub).item() * batch[0].shape[0]
                    val_seen += batch[0].shape[0]

            train_mse, val_mse = running / max(seen, 1), val_loss / max(val_seen, 1)
            history.append((epoch, train_mse, val_mse))
            print(f"epoch {epoch:3d}   train loss = {train_mse:.6f}   val loss = {val_mse:.6f}")
        return history
