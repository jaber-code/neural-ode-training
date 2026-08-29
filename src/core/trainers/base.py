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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from core.config import TrainConfig
from core.datasets.base import TransitionDataset
from core.distributed import dist_info
from core.integrators.base import Integrator


def _build_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    """LR schedule stepped once per epoch. "none" keeps lr fixed (the old
    behavior, and the default); "step"/"cosine" wrap the matching
    torch.optim.lr_scheduler class, reading its kwargs from whichever of
    step_scheduler_params/cosine_scheduler_params matches cfg.scheduler --
    both stay in the config at once, so switching between them only needs
    the one `scheduler` field to change."""
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, **cfg.step_scheduler_params)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, **cfg.cosine_scheduler_params)
    raise ValueError(f"unknown train.scheduler {cfg.scheduler!r} -- expected 'none', 'step', or 'cosine'")


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
        # cfg.distributed is the one on/off switch (see core/distributed.py): False always
        # runs this exact single-process path, regardless of how the process was launched.
        rank, world_size, local_rank = dist_info(cfg.distributed)
        distributed = world_size > 1
        if distributed:
            # SLURM's GPU binding may already restrict this process to seeing only its
            # one assigned GPU (as device 0), or it may leave all of them visible and
            # expect local_rank to pick the right index -- device_count() tells us which.
            gpu = local_rank if torch.cuda.device_count() > 1 else 0
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(gpu)
            device = f"cuda:{gpu}"  # one GPU per process, overriding whatever cfg.device says

        model.to(device)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[gpu])
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = _build_scheduler(opt, cfg)

        # Unchanged when not distributed: sampler=None, shuffle=True/False as before.
        # When distributed, DistributedSampler hands each rank a disjoint shard instead.
        train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
                                   sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_size=max(len(val_ds), 1), shuffle=False, sampler=val_sampler)

        history = []
        for epoch in range(cfg.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)  # reshuffles differently each epoch, same as shuffle=True would

            model.train()
            running, seen = 0.0, 0
            for batch in train_loader:
                batch = tuple(x.to(device) for x in batch)
                loss = self.compute_loss(model, integrator, batch, cfg.n_sub)
                opt.zero_grad()
                loss.backward()  # DDP all-reduces gradients across ranks here, transparently
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                running += loss.item() * batch[0].shape[0]
                seen += batch[0].shape[0]
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss, val_seen = 0.0, 0
                for batch in val_loader:
                    batch = tuple(x.to(device) for x in batch)
                    val_loss += self.compute_loss(model, integrator, batch, cfg.n_sub).item() * batch[0].shape[0]
                    val_seen += batch[0].shape[0]

            if distributed:
                # each rank only summed its own shard above -- combine into whole-dataset totals
                stats = torch.tensor([running, seen, val_loss, val_seen], dtype=torch.float64, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                running, seen, val_loss, val_seen = stats.tolist()

            train_mse, val_mse = running / max(seen, 1), val_loss / max(val_seen, 1)
            if rank == 0:  # every rank computed the same totals above; only log once
                history.append((epoch, train_mse, val_mse))
                print(f"epoch {epoch:3d}   train loss = {train_mse:.6f}   val loss = {val_mse:.6f}")

        if distributed:
            dist.destroy_process_group()
        return history
