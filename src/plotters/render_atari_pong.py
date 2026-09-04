"""Visualize one episode-chunk from the raw Atari-Pong ArrayRecord dataset:
saves its frames as a GIF with the recorded action index (and name, if a
Gymnasium env id is given) overlaid on each frame. A data sanity check, not
a model rollout -- no trained model/config involved, just the raw recorded
(frame, action) pairs read straight from one .array_record shard. Same
spirit as render_mujoco_rollout.py's visual checks, for the raw dataset
instead of a trained model.

Ported from p-doom/jasmine's own visualization script (shared by the repo
owner), adapted to auto-pick a real data shard under a dataset directory
instead of requiring an exact file path, and to this project's output
conventions.

Needs `array_record` and, for action names, `gymnasium[atari]`/`ale-py` --
`array_record` only ships Linux wheels, so this has to run on the server,
same as core/datasets/atari_pong.py's _debug_explore (see its docstring).

Usage:
    python plotters/render_atari_pong.py [--path datasets/atari_pong] [--env_id ALE/Pong-v5]
"""

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import numpy as np
from array_record.python.array_record_module import ArrayRecordReader
from PIL import Image, ImageDraw

DEFAULT_PATH = "/fscratch/sjaber/atari"
DEFAULT_ENV_ID = "ALE/Pong-v5"


def unique_path(path: Path) -> Path:
    """path if free, else path with _2, _3, ... inserted before the extension --
    whichever doesn't exist yet."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def find_shard(path: Path) -> Path:
    """One real .array_record data shard under `path`, skipping
    huggingface_hub's own .cache/ bookkeeping directory -- it sorts first
    alphabetically and isn't real data (see core/datasets/atari_pong.py's
    _debug_explore, which hit exactly this)."""
    shards = sorted(f for f in path.rglob("*.array_record") if ".cache" not in f.parts)
    if not shards:
        raise FileNotFoundError(f"no .array_record files found under {path}")
    return shards[0]


def infer_hw_from_bytes(total_elements: int, seq_len: int, channels: int) -> tuple[int, int]:
    assert seq_len > 0 and channels > 0, "sequence_length and channels must be positive"
    base = total_elements // (seq_len * channels)
    side = int(math.isqrt(base))
    if side * side != base:
        raise ValueError(
            f"Could not infer square HxW from buffer. elements={total_elements}, "
            f"seq_len={seq_len}, channels={channels}"
        )
    return side, side


def get_action_meanings(env_id: Optional[str]) -> Optional[List[str]]:
    if not env_id:
        return None
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        env = gym.make(env_id)
        try:
            return list(env.unwrapped.get_action_meanings())
        finally:
            env.close()
    except Exception as e:
        print(f"couldn't get action meanings for {env_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PATH, help="downloaded atari_pong dataset directory")
    parser.add_argument("--input", default=None,
                         help="a specific .array_record file (default: auto-pick one under --path)")
    parser.add_argument("--channels", type=int, default=3,
                         help="only 3 (RGB) is supported -- the source data is grayscale repeated into 3 channels")
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--env_id", default=DEFAULT_ENV_ID, help="Gymnasium env id for action-name labels, or '' to skip")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    assert args.channels == 3, "only 3 channels are currently supported"

    shard = Path(args.input) if args.input else find_shard(Path(args.path))
    print(f"reading {shard}")

    reader = ArrayRecordReader(str(shard))
    raw = reader.read()
    if raw is None:
        raise RuntimeError(f"no record could be read from {shard}")
    record = pickle.loads(raw)

    seq_len = int(record["sequence_length"])
    has_actions = "actions" in record and record["actions"] is not None
    print(f"record: sequence_length={seq_len}  actions_present={has_actions}")

    arr = np.frombuffer(record["raw_video"], dtype=np.uint8)
    total_elements = arr.size
    h, w = args.height, args.width
    if h and w and seq_len * h * w * args.channels != total_elements:
        print(f"--height/--width ({h}x{w}) don't match the buffer size "
              f"({total_elements} elements for seq_len={seq_len}); inferring instead")
        h, w = None, None
    if not (h and w):
        h, w = infer_hw_from_bytes(total_elements, seq_len, args.channels)
    frames = arr.reshape(seq_len, h, w, args.channels)

    actions = None
    if has_actions:
        actions = np.asarray(record["actions"]).reshape(-1)
        assert actions.shape[0] == seq_len, f"expected {seq_len} actions, got {actions.shape[0]}"

    action_meanings = get_action_meanings(args.env_id)

    out_dir = Path("output/renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gif = unique_path(out_dir / f"{shard.stem}_visualize.gif")

    duration_ms = max(1, int(1000 / max(1, args.fps)))
    imgs = []
    print(f"actions, frame by frame (the gif plays these too fast to read):")
    for t in range(seq_len):
        img = Image.fromarray(frames[t], mode="RGB")
        draw = ImageDraw.Draw(img)
        if actions is not None:
            a = int(actions[t])
            name = action_meanings[a] if action_meanings and 0 <= a < len(action_meanings) else None
            text = f"{a}" if name is None else f"{a} {name}"
        else:
            text = "?"
        print(f"  frame {t:3d}: {text}")
        draw.text((2, 2), text, fill=(255, 255, 255))
        imgs.append(img)

    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
    print(f"saved {out_gif}  ({seq_len} frames, {h}x{w}x{args.channels}, {args.fps} fps)")


if __name__ == "__main__":
    main()
