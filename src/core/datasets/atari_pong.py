"""Atari-Pong: https://huggingface.co/datasets/p-doom/atari-pong-dataset
10M frames, 84x84 grayscale (repeated to 3 channels), ArrayRecord format,
~753MB, from Rainbow-agent training -- a small first Atari dataset for
getting the pipeline working before considering something bigger (e.g. the
DQN Replay / "Dopamine" dataset, tens of GB per game per run).

Confirmed schema (from _debug_explore's output against the real downloaded
data, no longer guessed): each record is a pickled dict with
  raw_video:        bytes           -- flat buffer, reshapes to
                                        (sequence_length, H, W, 3) uint8
  sequence_length:  int             -- 160 per record here (matches Jasmine's
                                        own data-gen script's chunk_size)
  actions:          ndarray (T,) int8 -- ONE DISCRETE action index per frame,
                                        not a continuous vector
Real shard files live under <path>/<split>/*.array_record (confirmed via the
render script's auto-picked shard), 100 records/shard here.

Deliberately NOT eager-loading the whole 10M-frame dataset like
MuJoCoDataset does for Hopper -- at 84*84*3 bytes/frame that's >200GB
unpacked, and this dataset is orders of magnitude bigger than what the rest
of this codebase assumes fits in memory. Instead: read `max_records`
episode-chunks (default 100 -- a few hundred MB, enough to get the pipeline
running end to end) and hold those as uint8 (not float32) to keep the
memory footprint down; __getitem__ casts to float32 and normalizes to [0,1]
per access, not upfront.

Two things this does NOT resolve, both still open:
  - action_dim is set to 1 and the discrete action id is cast to a float --
    a placeholder that satisfies TransitionDataset's interface, not a real
    design choice. A discrete action likely wants an embedding lookup on the
    model side, not concatenation like Hopper's continuous torques.
  - state_dim is a flattened 84*84*3 frame (TransitionDataset's "1D float32
    state" contract), not a (H,W,C) image -- fine for now, but a CNN model
    will want to reshape it back via self.frame_shape.
supports_windows is left False: rows from different concatenated episode-
chunks aren't temporally adjacent, and unlike MuJoCoDataset there's no
terminal/timeout column here yet to mark those boundaries -- multi_step
training would silently window across unrelated episodes if this were True.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import torch

from core.registry import DATASETS

from .base import TransitionDataset

REPO_ID = "p-doom/atari-pong-dataset"


@DATASETS.register("atari_pong")
class AtariPongDataset(TransitionDataset):
    def __init__(self, path: str, dt: float, split: str = "test", max_records: int = 100):
        dest = Path(path)
        self._ensure_downloaded(dest)

        from array_record.python.array_record_module import ArrayRecordReader

        split_dir = dest / split
        shards = sorted(f for f in split_dir.rglob("*.array_record") if ".cache" not in f.parts)
        if not shards:
            shards = sorted(f for f in dest.rglob("*.array_record") if ".cache" not in f.parts)
        if not shards:
            raise FileNotFoundError(f"no .array_record files found under {dest} (split={split!r})")

        frames_per_record, actions_per_record = [], []
        n_loaded = 0
        for shard in shards:
            if n_loaded >= max_records:
                break
            reader = ArrayRecordReader(str(shard))
            for raw in reader.read_all():
                if n_loaded >= max_records:
                    break
                record = pickle.loads(raw)
                seq_len = int(record["sequence_length"])
                acts = record.get("actions")
                if acts is None:
                    continue  # schema marks this optional; skip episodes that lack it
                acts = np.asarray(acts).reshape(-1)
                if acts.shape[0] != seq_len:
                    continue  # malformed/truncated record -- skip rather than guess

                arr = np.frombuffer(record["raw_video"], dtype=np.uint8)
                h, w = self._infer_hw(arr.size, seq_len, channels=3)
                frames_per_record.append(arr.reshape(seq_len, h, w, 3))
                actions_per_record.append(acts)
                n_loaded += 1

        if not frames_per_record:
            raise RuntimeError(f"no usable records loaded from {split_dir} (checked {len(shards)} shard(s))")
        print(f"atari_pong: loaded {n_loaded} episode-chunk(s) from {len(shards)} shard(s) under {split_dir}")

        # pair up consecutive frames WITHIN each chunk into (s1, a, s2) transitions --
        # never across chunks, since two different chunks aren't temporally adjacent
        s1_chunks, a_chunks, s2_chunks = [], [], []
        for frames, acts in zip(frames_per_record, actions_per_record):
            s1_chunks.append(frames[:-1])
            s2_chunks.append(frames[1:])
            a_chunks.append(acts[:-1])  # action taken AT s1, producing s2

        s1 = np.concatenate(s1_chunks, axis=0)  # (N, H, W, 3) uint8
        s2 = np.concatenate(s2_chunks, axis=0)
        a = np.concatenate(a_chunks, axis=0)    # (N,) int8, discrete action index

        self.frame_shape = s1.shape[1:]  # (H, W, 3) -- for a future CNN model to reshape back from flat
        self.state_dim = int(np.prod(self.frame_shape))
        self.action_dim = 1  # placeholder: a discrete index cast to float, see module docstring
        self.dt = float(dt)

        # kept as uint8/int64, not float32 -- __getitem__ converts per access (see module docstring)
        self.s1 = torch.from_numpy(s1.reshape(len(s1), -1))
        self.s2 = torch.from_numpy(s2.reshape(len(s2), -1))
        self.a = torch.from_numpy(a.astype(np.int64))

    @staticmethod
    def _infer_hw(total_elements: int, seq_len: int, channels: int = 3) -> tuple[int, int]:
        """raw_video has no explicit width/height field -- infer a square frame
        size from the flat buffer length (matches render_atari_pong.py's
        approach, which confirmed 84x84x3 against the real data)."""
        base = total_elements // (seq_len * channels)
        side = int(math.isqrt(base))
        if side * side != base:
            raise ValueError(
                f"can't infer a square HxW from raw_video "
                f"(elements={total_elements}, seq_len={seq_len}, channels={channels})"
            )
        return side, side

    @staticmethod
    def _ensure_downloaded(dest: Path) -> None:
        """Fetch the dataset from HuggingFace into `dest` if it's not already
        there. Uses the `huggingface_hub` package directly (not the
        `huggingface-cli` shell command, which needs its pip --user bin dir
        on PATH -- the Python import doesn't). Safe to call redundantly/
        concurrently -- e.g. one call per rank under distributed training --
        since snapshot_download uses per-file lock files, so simultaneous
        callers don't race each other."""
        if dest.is_dir() and any(dest.iterdir()):
            return  # already downloaded
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "downloading the atari_pong dataset needs huggingface_hub: pip install --user huggingface_hub"
            ) from e
        print(f"downloading {REPO_ID} to {dest} ...")
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(dest))
        print(f"done: {dest}")

    def __len__(self) -> int:
        return self.s1.shape[0]

    def __getitem__(self, idx):
        dt = torch.tensor([self.dt], dtype=torch.float32)
        s1 = self.s1[idx].to(torch.float32) / 255.0
        s2 = self.s2[idx].to(torch.float32) / 255.0
        a = self.a[idx].to(torch.float32).unsqueeze(-1)  # (1,) -- see action_dim note in module docstring
        return s1, a, dt, s2
