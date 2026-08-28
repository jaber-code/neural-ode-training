"""
The SIMPLE, satisfying version: predict s2 directly from (s1, a) at the fixed
training dt. Walls included. No rollout, no integration -> no soft-wall problem.

This is a discrete next-state predictor. It is NOT a continuous-time model
(that's the later question). But it WILL work cleanly, walls and all -- which
is the point: confirm the toy is learnable, then move on.
"""

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# Load data and split into train / validation
#   each row: s1x s1y  ax ay  dt  s2x s2y
# ----------------------------------------------------------------------
data = torch.tensor(np.load("../../../data/gridball_vardt.npy"), dtype=torch.float32)

n = data.shape[0]
perm = torch.randperm(n)
n_val = n // 10
val_idx = perm[:n_val]
train_idx = perm[n_val:]

def get(idx):
    """Return (inputs, targets) for the given rows.
       inputs  X = (s1, a)   -> 4 numbers
       targets y = s2        -> 2 numbers
    """
    X = torch.cat([data[idx, 0:2], data[idx, 2:4]], dim=1)
    #y = data[idx, 5:7]
    # becomes (vector field):
    s1 = data[idx, 0:2]
    s2 = data[idx, 5:7]
    dt = data[idx, 4:5]
    y = (s2 - s1) / dt      

    return X, y

X_train, y_train = get(train_idx)
X_val,   y_val   = get(val_idx)

# ----------------------------------------------------------------------
# Model, optimizer, loss
# ----------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

"""
=======================================================
overall  mean endpoint error : 0.0041
interior mean endpoint error : 0.0039
wall     mean endpoint error : 0.0043
worst-case endpoint error    : 0.0193
"""
# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
BATCH = 128
EPOCHS = 50

for epoch in range(EPOCHS):
    order = torch.randperm(X_train.shape[0])
    running = 0.0

    for i in range(0, X_train.shape[0], BATCH):
        idx = order[i:i + BATCH]

        pred = model(X_train[idx])

        loss = loss_fn(pred, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item() * len(idx)

    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()

    train_loss = running / X_train.shape[0]
    print(f"epoch {epoch:2d}   train MSE = {train_loss:.6f}   val MSE = {val_loss:.6f}")


# ----------------------------------------------------------------------
# Evaluation: prediction error, split by interior vs wall
# ----------------------------------------------------------------------
print("\n" + "=" * 55)

with torch.no_grad():
    S1, A, DT, S2 = data[:, 0:2], data[:, 2:4], data[:, 4:5], data[:, 5:7]

    #1
    #pred = model(torch.cat([S1, A], dim=1))
    
    #2 inference: one Euler step at dt = 0.2
    #v = model(torch.cat([S1, A], dim=1))
    #pred = S1 + v * DT
    #3
    TEST_DT = 0.3
    v = model(torch.cat([S1, A], dim=1))
    pred = S1 + v * TEST_DT
    #err = (pred - S2).norm(dim=1)
    true_01 = torch.clamp(S1 + A * TEST_DT, 0.0, 1.0)   # exact next state at dt=0.1
    err = (pred - true_01).norm(dim=1)
    # a row "hits a wall" if the un-clipped landing leaves the box
    free = S1 + A * TEST_DT
    wall = torch.any((free < 0) | (free > 1), dim=1)

    print(f"overall  mean endpoint error : {err.mean():.4f}")
    print(f"interior mean endpoint error : {err[~wall].mean():.4f}")
    print(f"wall     mean endpoint error : {err[wall].mean():.4f}")
    print(f"worst-case endpoint error    : {err.max():.4f}")

    # predictions should stay inside the box if walls were learned
    inside = ((pred >= -0.02) & (pred <= 1.02)).all(dim=1).float().mean()
    print(f"fraction of predictions inside the box: {inside * 100:.1f}%")


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
torch.save(model.state_dict(), "../../../output/vfield_onestep.pt")
print("saved ../../../output/vfield_onestep.pt")