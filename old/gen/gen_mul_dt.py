"""
Step-1 toy with VARIABLE dt: 2D velocity-controlled ball in a box, stopping walls.

Same dynamics as before -- s2 = clip(s1 + a*dt) -- but now each row has its OWN
randomly drawn dt instead of a fixed 0.2. This forces the learned field to be
correct across MANY durations at once, so it can't bake in a single-dt distance
(the bug that made predictions wrong at dt != 0.2).

Each row of the saved data is:  s1x s1y  ax ay  dt  s2x s2y
"""

import numpy as np

# --- config (change these) ---------------------------------------------
SEED   = 0
N      = 50_000
DT_MIN = 0.05          # dt is now a RANGE, drawn per row
DT_MAX = 0.3
BOT_LEFT, HIGH_TOP = np.array([0., 0.]), np.array([1., 1.])
A_MAX  = 3.0           # velocity range: [-A_MAX, A_MAX]

rng = np.random.default_rng(SEED)

# --- the environment: one function, reuse it for eval later ------------
def step(s1, a, dt, lo=BOT_LEFT, hi=HIGH_TOP):
    return np.clip(s1 + a * dt, lo, hi)

# --- generate ----------------------------------------------------------
S1 = rng.uniform(BOT_LEFT, HIGH_TOP, size=(N, 2))     # start positions
A  = rng.uniform(-A_MAX, A_MAX, size=(N, 2))          # velocity commands
DT = rng.uniform(DT_MIN, DT_MAX, size=(N, 1))         # <-- per-row random dt
#DT = rng.choice([0.05, 0.1, 0.2, 0.3], size=(N, 1))

S2 = step(S1, A, DT)                                  # next states (uses each row's dt)

data = np.concatenate([S1, A, DT, S2], axis=1)        # DT is already (N,1)
np.save("datasets/2d_grid_toy/gridball_contDT.npy", data)
np.savetxt("datasets/2d_grid_toy/gridball_contDT.csv", data, delimiter=",",
           header="s1x,s1y,ax,ay,dt,s2x,s2y", comments="")

# --- quick sanity check ------------------------------------------------
free = S1 + A * DT
wall = np.any((free < BOT_LEFT) | (free > HIGH_TOP), axis=1)
print(f"saved gridball_vardt.npy  shape={data.shape}")
print(f"dt range: {DT.min():.3f} .. {DT.max():.3f}   mean {DT.mean():.3f}")
print(f"wall-hit fraction: {wall.mean():.3f}")