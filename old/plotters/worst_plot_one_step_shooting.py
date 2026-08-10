"""
Top-50 worst cases for a SHOOTING model, evaluated the way it was trained:
by INTEGRATING the field over TEST_DT in sub-steps (not one Euler step).

  black dot = start   green sq = true next @ TEST_DT   red x = predicted   orange = action
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

TEST_DT = 0.1
N_WORST = 50
N_SUB   = 8            # MUST match the sub-steps used in training

# architecture MUST match what you trained (2 hidden layers)
model = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 2),
)
model.load_state_dict(torch.load("output/vfield_shooting_con_dt.pt"))   # the shooting model
model.eval()

data = torch.tensor(np.load("data/gridball_contDT.npy"), dtype=torch.float32)
S1, A = data[:, 0:2], data[:, 2:4]

# integrate the field over TEST_DT in sub-steps (same as training)
def integrate(s0, a, dt, n_sub=N_SUB):
    h = dt / n_sub
    s = s0
    with torch.no_grad():
        for _ in range(n_sub):
            s = s + model(torch.cat([s, a], dim=1)) * h
    return s

dt_col = torch.full((S1.shape[0], 1), TEST_DT)
pred = integrate(S1, A, dt_col)
true = torch.clamp(S1 + A * TEST_DT, 0.0, 1.0)
err = (pred - true).norm(dim=1)

worst = torch.argsort(err, descending=True)[:N_WORST]
print(f"[dt={TEST_DT}] worst {N_WORST}: {err[worst[-1]]:.4f} .. {err[worst[0]]:.4f}")
print(f"[dt={TEST_DT}] median error : {err.median():.4f}")

S1n, An, truen, predn, errn = S1.numpy(), A.numpy(), true.numpy(), pred.numpy(), err.numpy()

cols = 10
rows = (N_WORST + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(1.9*cols, 1.9*rows))
axes = axes.ravel()
for k, i in enumerate(worst.numpy()):
    ax = axes[k]
    s1, s2, p, a = S1n[i], truen[i], predn[i], An[i]
    ax.add_patch(plt.Rectangle((0,0),1,1, fill=False, ec="gray", lw=1)) # type: ignore
    an = a/(np.linalg.norm(a)+1e-9)*0.18
    ax.arrow(s1[0],s1[1],an[0],an[1], color="orange", width=0.01,
             head_width=0.06, length_includes_head=True, alpha=0.7)
    ax.plot(*s1,"ko",ms=5); ax.plot(*s2,"s",color="green",ms=7,alpha=0.8)
    ax.plot(*p,"x",color="red",ms=7,mew=2)
    ax.plot([s2[0],p[0]],[s2[1],p[1]],"r:",lw=1)
    ax.set_title(f"{errn[i]:.3f}",fontsize=8)
    ax.set_xlim(-0.2,1.2);ax.set_ylim(-0.2,1.2);ax.set_aspect("equal")
    ax.set_xticks([]);ax.set_yticks([])
for k in range(N_WORST,len(axes)): axes[k].axis("off")
fig.suptitle(f"Top {N_WORST} worst (SHOOTING model, integrated) at dt={TEST_DT}\n"
             "black=start  green=true  red=predicted  orange=action",fontsize=12,y=1.01)
fig.tight_layout()
fig.savefig(f"worst50_shooting_condt{TEST_DT}.png", dpi=120, bbox_inches="tight")
print(f"saved worst50_shooting_condt{TEST_DT}.png")