"""
Step-1 toy: 2D velocity-controlled ball in a box with stopping walls.

Ground-truth dynamics (exact for this system):
    s2 = clip(s1 + a * dt)      # move at commanded velocity, stop at walls

Each row of the saved data is:  s1x s1y  ax ay  dt  s2x s2y
"""

import numpy as np

# --- config (change these) ---------------------------------------------
SEED   = 0
N      = 50_000
DT     = 0.2
BOT_LEFT, HIGH_TOP = np.array([0., 0.]), np.array([1., 1.]) 
A_MAX  = 3.0 # velocity range: [-A_MAX, A_MAX]

rng = np.random.default_rng(SEED)

# --- the environment: one function, reuse it for eval later ------------
def step(s1, a, dt, lo=BOT_LEFT, hi=HIGH_TOP):
    return np.clip(s1 + a * dt, lo, hi)

# --- generate ----------------------------------------------------------
S1 = rng.uniform(BOT_LEFT, HIGH_TOP, size=(N, 2))  # start positions
A  = rng.uniform(-A_MAX, A_MAX, size=(N, 2))       # velocity commands
S2 = step(S1, A, DT)                               # next states

data = np.concatenate([S1, A, np.full((N, 1), DT), S2], axis=1)
np.save(f"../../../data/gridball_fixedDT_{DT}.npy", data)
np.savetxt(f"../../../data/gridball_fixedDT_{DT}.csv", data, delimiter=",", header="s1x,s1y,ax,ay,dt,s2x,s2y", comments="")

# --- quick sanity check ------------------------------------------------
wall = np.any((S1 + A*DT < BOT_LEFT) | (S1 + A*DT > HIGH_TOP), axis=1)
print(f"saved gridball.npy  shape={data.shape}  wall-hit fraction={wall.mean():.3f}")