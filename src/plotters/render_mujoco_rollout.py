"""Render a trained MuJoCo vector-field model's predicted rollout side by
side with the true recorded rollout, as an animated GIF. This is the Hopper
equivalent of the gridball worst-case plots: instead of comparing two 2D
points, we render the full robot pose at every step and compare motion.

Method: take a real contiguous window of (s1, a) from the dataset, replay
the SAME recorded action sequence through the trained model+integrator
(instead of the true simulator) to get a predicted state trajectory, and
render both that and the true recorded trajectory with MuJoCo's own
renderer via env.unwrapped.set_state(qpos, qvel). The XML model itself
(Hopper's skeleton) never changes -- only the qpos/qvel we feed it each
frame differs between the "true" and "model" pass.

The two sides play at DIFFERENT resolutions on purpose. Left (true) only
has real recorded points every dt=0.008s, so it just holds each one until
the next real point arrives. Right (model) shows every one of the n_sub
internal baby-steps the integrator takes to get from one real point to the
next -- this is the actual thesis question: what does the model think
happens BETWEEN two real observations, not just at them. Left updates once
per real tick; right updates every baby-step, so you can watch whether the
model's in-between motion looks physically plausible, and whether it lands
back near the real point each time left jumps to catch up.

The model only sees the 11-dim observation, which excludes the robot's
absolute x-position (dropped during training on purpose, for translation
invariance) -- but it does include x-velocity, so x-position is
reconstructed here by integrating that, purely for rendering.

Usage:
    python plotters/render_mujoco_rollout.py [config path]
    (defaults to configs/step2_mujoco_euler.yaml if omitted)

Every run's outputs land in one shared output/renders/ folder, named after
the config that produced them (output/renders/<config name>_rollout.gif,
output/renders/<config name>_error.png), with _2, _3, ... appended if that
name was already used -- so results from different configs, or repeated
runs of the same one, never overwrite each other.
"""

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import core.datasets  # noqa: F401
import core.integrators  # noqa: F401
import core.models  # noqa: F401
from core.config import load_config
from core.datasets.mujoco import MuJoCoDataset
from core.registry import DATASETS, INTEGRATORS, MODELS

DEFAULT_CONFIG_PATH = "configs/step2_mujoco_euler_multistep.yaml"
START_IDX = 500            # row to start the window at (skip the very first few resets)
N_STEPS = 150               # rollout length in control steps
FPS = 25                    # env's real control rate is 1/dt = 125Hz; slowed down for visibility

# Hopper-v4's own termination criterion (env.unwrapped._healthy_z_range /
# _healthy_angle_range) -- "fell over" means leaving these, not a number we chose.
HEALTHY_Z_RANGE = (0.7, float("inf"))
HEALTHY_ANGLE_RANGE = (-0.2, 0.2)


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


def find_contiguous_window(dataset, start_idx: int, want_len: int):
    """Consecutive rows in the (filtered) dataset aren't guaranteed to be
    consecutive real timesteps -- the transition at an episode boundary was
    dropped, so the next surviving row may belong to a different episode.
    Detect that by checking s2[i] matches s1[i+1] (true for same-episode
    neighbors, since next_obs came directly from the recording)."""
    end = start_idx
    while end + 1 < len(dataset) and end - start_idx < want_len:
        if not torch.allclose(dataset.s1[end + 1], dataset.s2[end], atol=1e-5):
            break
        end += 1
    return start_idx, end  # inclusive range [start_idx, end]


def unnormalize(dataset, obs: torch.Tensor) -> torch.Tensor:
    if not hasattr(dataset, "obs_mean"):
        return obs
    mean = torch.as_tensor(dataset.obs_mean, dtype=obs.dtype)
    std = torch.as_tensor(dataset.obs_std, dtype=obs.dtype)
    return obs * std + mean


def obs_to_qpos_qvel(obs_row: np.ndarray, x_pos: float):
    """11-dim Hopper observation -> full 6-dim qpos/qvel MuJoCo needs.
    layout: [rootz, rooty, thigh, leg, foot,  rootx_vel, rootz_vel, rooty_vel, thigh_vel, leg_vel, foot_vel]"""
    qpos = np.array([x_pos, *obs_row[0:5]], dtype=np.float64)
    qvel = np.array(obs_row[5:11], dtype=np.float64)
    return qpos, qvel


def is_healthy(obs_row: np.ndarray) -> bool:
    z, angle = obs_row[0], obs_row[1]
    return HEALTHY_Z_RANGE[0] < z < HEALTHY_Z_RANGE[1] and HEALTHY_ANGLE_RANGE[0] < angle < HEALTHY_ANGLE_RANGE[1]


def first_unhealthy_step(obs_seq: np.ndarray):
    for t, row in enumerate(obs_seq):
        if not is_healthy(row):
            return t
    return None  # stayed healthy for the whole window


def measure_rollout(true_obs: np.ndarray, pred_obs: np.ndarray, out_png: Path):
    """Two formal (non-visual) checks:
      1. per-step state error, to see whether the closed-loop rollout drifts
         away from the true trajectory gradually or blows up.
      2. does the model's implied robot ever leave the same "healthy" pose
         range the real Hopper-v4 env uses to end an episode -- i.e. did it
         fall over in a way the real trajectory didn't."""
    err = np.linalg.norm(pred_obs - true_obs, axis=1)  # per-step L2 error, real units
    true_fall = first_unhealthy_step(true_obs)
    pred_fall = first_unhealthy_step(pred_obs)

    print(f"per-step state error: mean={err.mean():.4f}  max={err.max():.4f}  final={err[-1]:.4f}")
    print(f"true  trajectory falls over (leaves healthy z/angle range) at step: "
          f"{true_fall if true_fall is not None else 'never in this window'}")
    print(f"model trajectory falls over at step: "
          f"{pred_fall if pred_fall is not None else 'never in this window'}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(err, color="crimson")
    if true_fall is not None:
        ax.axvline(true_fall, color="gray", ls="--", label=f"true falls @ step {true_fall}")
    if pred_fall is not None:
        ax.axvline(pred_fall, color="crimson", ls="--", label=f"model falls @ step {pred_fall}")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("||pred - true state||  (real units)")
    ax.set_title("closed-loop rollout: state error over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


def rollout_frames(env, obs_seq: np.ndarray, dt: float) -> list:
    frames = []
    x_pos = 0.0
    for t, obs_row in enumerate(obs_seq):
        if t > 0:
            x_pos += obs_row[5] * dt  # integrate x-velocity to reconstruct x-position
        qpos, qvel = obs_to_qpos_qvel(obs_row, x_pos)
        env.unwrapped.set_state(qpos, qvel)
        frames.append(env.render())
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH, help="path to a YAML experiment config")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = Path("output/renders")
    out_dir.mkdir(parents=True, exist_ok=True)
    # named after the checkpoint (not cfg.name) so a render is traceable back to the
    # exact trained run it came from -- train/run.py suffixes checkpoints with a run id
    checkpoint_stem = Path(cfg.checkpoint).stem
    out_gif = unique_path(out_dir / f"{checkpoint_stem}_rollout.gif")
    out_error_png = unique_path(out_dir / f"{checkpoint_stem}_error.png")

    dataset = cast(MuJoCoDataset, DATASETS.build(cfg.dataset.name, **cfg.dataset.params))
    model = MODELS.build(cfg.model.name, state_dim=dataset.state_dim, action_dim=dataset.action_dim, **cfg.model.params)
    integrator = INTEGRATORS.build(cfg.integrator.name, **cfg.integrator.params)

    # fall back to CPU when the config says "cuda" but this machine has no working
    # CUDA build of torch (e.g. a config written for the cluster, run locally)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
    model.eval()

    start, end = find_contiguous_window(dataset, START_IDX, N_STEPS)
    n = end - start
    note = "  (window ended early: hit an episode boundary)" if n < N_STEPS else ""
    print(f"using rows [{start}, {end}]  ({n} steps){note}")

    s1 = dataset.s1[start]
    actions = dataset.a[start:end]        # (n, action_dim), the REAL recorded actions
    true_obs = dataset.s1[start:end + 1]  # (n+1, state_dim), the REAL recorded states

    # replay the same actions through the trained model instead of the true simulator,
    # keeping every one of the n_sub internal baby-steps per hop (not just each hop's
    # final answer) so we can see what the model predicts BETWEEN the real recorded points
    n_sub = cfg.train.n_sub
    pred_obs = [s1]
    s = s1.unsqueeze(0)
    dt_col = torch.full((1, 1), dataset.dt)
    with torch.no_grad():
        for t in range(n):
            a = actions[t].unsqueeze(0)
            states = integrator.rollout(model, s, a, dt_col, n_sub, cfg.eval.perturb_action)  # (1+n_sub, state_dim)
            pred_obs.extend(x.squeeze(0) for x in states[1:])  # states[0] == s, already have it
            s = states[-1]
    pred_obs = torch.stack(pred_obs)  # 1 + n*n_sub states (vs. true_obs's 1 + n)

    true_obs_np = unnormalize(dataset, true_obs).numpy()
    pred_obs_np = unnormalize(dataset, pred_obs).numpy()

    # measure_rollout compares real checkpoints only -- pull out just the hop-boundary
    # states (every n_sub-th one), matching true_obs_np's length, same as before this change
    pred_obs_at_checkpoints = pred_obs_np[::n_sub]
    measure_rollout(true_obs_np, pred_obs_at_checkpoints, out_error_png)

    env = gym.make("Hopper-v4", render_mode="rgb_array")
    env.reset()
    true_frames = rollout_frames(env, true_obs_np, dataset.dt)      # one frame per real 0.008s tick
    h = dataset.dt / n_sub
    pred_frames = rollout_frames(env, pred_obs_np, h)               # one frame per baby-step
    env.close()

    # left holds the most recent real checkpoint while right plays through its baby
    # steps, then jumps to the next real checkpoint exactly when right reaches it
    combined = [
        Image.fromarray(np.concatenate([true_frames[k // n_sub], pred_frames[k]], axis=1))
        for k in range(len(pred_frames))
    ]
    combined[0].save(
        out_gif, save_all=True, append_images=combined[1:], duration=int(1000 / FPS), loop=0
    )
    print(f"saved {out_gif}  ({len(combined)} frames, left=true (every {n_sub}th frame) right=model (every baby-step))")


if __name__ == "__main__":
    main()
