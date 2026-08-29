"""Minimal torch.distributed helpers, shared by the trainer and the
entrypoint.

Distributed is opt-in via `train.distributed` in the config -- not
auto-detected -- so a plain `python train/run.py config.yaml` always runs
single-process regardless of environment, and there's one obvious place to
switch it off again.

Two launch mechanisms are supported, checked in this order:
  1. SLURM-native: `srun --ntasks=N ...` launches N processes directly (see
     run.slurm.sh), each seeing SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID.
     torch.distributed still needs MASTER_ADDR/MASTER_PORT, which SLURM
     doesn't set -- filled in here for the single-node case (localhost +
     a port derived from the job id, so concurrent jobs on the same node
     don't collide on the same port).
  2. torchrun: `torchrun --standalone --nproc_per_node=N ...` sets
     RANK/WORLD_SIZE/LOCAL_RANK itself and handles MASTER_ADDR/PORT
     internally -- useful for testing multi-GPU outside of SLURM.
"""

import os


def dist_info(enabled: bool) -> tuple[int, int, int]:
    """(rank, world_size, local_rank).

    Returns (0, 1, 0) -- i.e. "not distributed" -- whenever `enabled` is
    False, regardless of how the process was launched, so
    `train.distributed: false` always guarantees a plain single-process run
    (e.g. for local runs on a machine with no multi-GPU setup at all).

    When `enabled` is True, reads identity from whichever launcher's env
    vars are present (SLURM's own, or torchrun's), and raises a clear error
    if neither is -- i.e. distributed: true but launched with plain
    `python`.
    """
    if not enabled:
        return 0, 1, 0
    if "SLURM_PROCID" in os.environ:
        os.environ.setdefault("MASTER_ADDR", "localhost")  # single-node only, per current run.slurm.sh
        os.environ.setdefault("MASTER_PORT", str(20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 20000))
        return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    raise RuntimeError(
        "train.distributed is true but this process wasn't launched via `srun --ntasks=N` "
        "(SLURM_PROCID isn't set) or `torchrun` (WORLD_SIZE isn't set). "
        "See run.slurm.sh for the SLURM launch, or set train.distributed: false to run single-process."
    )


def is_main_process(enabled: bool) -> bool:
    """True for the single process in the non-distributed case, and for rank 0
    when distributed -- guards work that should only happen once (printing,
    eval sweeps, checkpoint saving), not once per GPU."""
    return dist_info(enabled)[0] == 0
