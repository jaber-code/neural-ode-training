
import numpy as np
import torch
import torch.nn as nn

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

N_SUB = 8          # integration sub-steps during training

# ---- data + split -----------------------------------------------------
data = torch.tensor(np.load("datasets/2d_grid_toy/gridball_contDT.npy"), dtype=torch.float32)
n = data.shape[0]
perm = torch.randperm(n)
n_val = n // 10
val_idx, train_idx = perm[:n_val], perm[n_val:]

def get(idx):
    s1 = data[idx, 0:2]
    a  = data[idx, 2:4]
    dt = data[idx, 4:5]
    s2 = data[idx, 5:7]
    return s1, a, dt, s2

S1tr, Atr, DTtr, S2tr = get(train_idx)
S1va, Ava, DTva, S2va = get(val_idx)

# ---- model (2 hidden layers, matching your current setup) -------------
model = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---- shooting: integrate the field over dt (NO clipping) --------------
def integrate(s0, a, dt, n_sub=N_SUB):
    h = dt / n_sub                      # dt is (B,1), so h is per-row
    s = s0
    for _ in range(n_sub):
        v = model(torch.cat([s, a], dim=1))
        s = s + v * h
    return s

def shooting_loss(s1, a, dt, s2):
    pred = integrate(s1, a, dt)
    return loss_fn(pred, s2)

# ---- train ------------------------------------------------------------
BATCH = 128
EPOCHS = 30

for epoch in range(EPOCHS):
    order = torch.randperm(S1tr.shape[0])
    running = 0.0
    for i in range(0, S1tr.shape[0], BATCH):
        idx = order[i:i+BATCH]
        loss = shooting_loss(S1tr[idx], Atr[idx], DTtr[idx], S2tr[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running += loss.item() * len(idx)
    with torch.no_grad():
        val = shooting_loss(S1va, Ava, DTva, S2va).item()
    print(f"epoch {epoch:2d}   train endpoint-MSE = {running/S1tr.shape[0]:.6f}   val = {val:.6f}")

# ---- evaluation: held-out dt test -------------------------------------
# The real test: pick dt values and check the field, INCLUDING a dt the
# training never emphasized. Compare model rollout vs true dynamics.
print("\n" + "=" * 55)
S1, A = data[:, 0:2], data[:, 2:4]

def rollout(s0, a, dt, n_sub=N_SUB):
    h = dt / n_sub
    s = s0
    with torch.no_grad():
        for _ in range(n_sub):
            s = s + model(torch.cat([s, a], dim=1)) * h
    return s

for test_dt in [0.1, 0.15, 0.2, 0.3, 0.4]:
    dt_col = torch.full((S1.shape[0], 1), test_dt)
    pred = rollout(S1, A, dt_col)
    true = torch.clamp(S1 + A * test_dt, 0.0, 1.0)
    err = (pred - true).norm(dim=1)
    free = S1 + A * test_dt
    wall = torch.any((free < 0) | (free > 1), dim=1)
    tag = "  (in range)" if 0.05 <= test_dt <= 0.30 else "  (EXTRAPOLATION)"
    print(f"dt={test_dt:<4}  overall={err.mean():.4f}  interior={err[~wall].mean():.4f}  "
          f"wall={err[wall].mean():.4f}{tag}")

torch.save(model.state_dict(), "output/vfield_shooting_con_dt.pt")
print("\nsaved output/vfield_shooting_con_dt.pt")