"""
Step-1 trainer with the SHOOTING target (the fix for the FD wall bias).

Difference from train_min.py: instead of regressing v onto (s2-s1)/dt,
we integrate the learned field forward from s1 under action a for the full
dt (in a few sub-steps) and match the ENDPOINT s2. Backprop flows through
the integration. This lets the field be sharp at the wall (full speed, then
stop) instead of the FD-averaged sag.

Same model, same data, same 4-test eval -> directly comparable to train_min.py.
"""

import numpy as np
import torch
import torch.nn as nn

# reproducibility: fix all RNG so runs are comparable (change only ONE thing at a time)
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

LO, HI  = 0.0, 1.0
A_MAX   = 3.0
DT_DATA = 0.2
N_SUB   = 8            # integration sub-steps during training (h = dt/N_SUB = 0.025)

# ---- data + split -----------------------------------------------------
data = torch.tensor(np.load("../../../gridball.npy"), dtype=torch.float32)
n = data.shape[0]; perm = torch.randperm(n); nv = n//10
va, tr = perm[:nv], perm[nv:]
def get(idx):
    return data[idx,0:2], data[idx,2:4], data[idx,4:5], data[idx,5:7]  # s1,a,dt,s2
S1tr,Atr,DTtr,S2tr = get(tr)
S1va,Ava,DTva,S2va = get(va)

# ---- model ------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(4,128), nn.ReLU(),
    nn.Linear(128,128), nn.ReLU(),
    nn.Linear(128,2),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- the shooting rollout (integrate learned field, NO clipping) ------
def integrate(s0, a, dt, n_sub=N_SUB):
    h = dt / n_sub                       # dt is (B,1); h is (B,1)
    s = s0
    for _ in range(n_sub):
        v = model(torch.cat([s, a], dim=1))
        s = s + v * h
    return s

def shooting_loss(S1,A,DT,S2):
    pred = integrate(S1, A, DT)
    return ((pred - S2)**2).mean()

# ---- train ------------------------------------------------------------
batch = 512
for epoch in range(30):
    p = torch.randperm(S1tr.shape[0]); tot=0.0
    for i in range(0, S1tr.shape[0], batch):
        idx = p[i:i+batch]
        loss = shooting_loss(S1tr[idx],Atr[idx],DTtr[idx],S2tr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # safety
        opt.step(); tot += loss.item()*len(idx)
    with torch.no_grad():
        val = shooting_loss(S1va,Ava,DTva,S2va).item()
    print(f"epoch {epoch:2d}  train endpoint-MSE = {tot/S1tr.shape[0]:.6f}   val = {val:.6f}")

# ======================================================================
# SAME 4-test battery as before
# ======================================================================
print("\n"+"="*60)

# 1. clean interior (mid-box + small actions -> true field = a)
with torch.no_grad():
    s = torch.rand(5000,2)*0.4+0.3
    a = (torch.rand(5000,2)*2-1)*(A_MAX*0.3)
    err_int = (model(torch.cat([s,a],1))-a).abs().mean().item()
print(f"1. CLEAN interior error |v - a|      = {err_int:.4f}   (want ~0)")

# 2. wall boundary (on right wall, pushing in -> true (0, a_y))
with torch.no_grad():
    m=500
    s=torch.stack([torch.ones(m),torch.rand(m)],1)
    a=torch.stack([torch.rand(m)*A_MAX,(torch.rand(m)*2-1)*A_MAX],1)
    v=model(torch.cat([s,a],1))
    tv=torch.stack([torch.zeros(m),a[:,1]],1)
    print(f"2. WALL boundary error |v - v_true|  = {(v-tv).abs().mean().item():.4f}   (want ~0)")
    print(f"   (into-wall component, want 0)     = {v[:,0].abs().mean().item():.4f}")

# 3. step-size rollout into right wall (true final = (1.0, 0.5))
def rollout(s0,a,T,h):
    s=s0.clone()
    with torch.no_grad():
        for _ in range(int(round(T/h))):
            s=s+model(torch.cat([s,a],1))*h
    return s
s0=torch.tensor([[0.5,0.5]]); a=torch.tensor([[3.0,0.0]]); T=0.5
tf=torch.clamp(s0+a*T,LO,HI)
print(f"3. STEP-SIZE rollout (true final = {tf.numpy().round(3).tolist()})")
for h in [0.2,0.1,0.05,0.01]:
    sf=rollout(s0,a,T,h); e=(sf-tf).abs().max().item()
    print(f"   h={h:<5} final={sf.numpy().round(3).tolist()}  err={e:.4f}")

# 4. field compare csv
with torch.no_grad():
    g=15; xs=torch.linspace(0,1,g); gx,gy=torch.meshgrid(xs,xs,indexing='xy')
    pos=torch.stack([gx.reshape(-1),gy.reshape(-1)],1)
    act=torch.tensor([[2.0,0.0]]).repeat(pos.shape[0],1)
    vl=model(torch.cat([pos,act],1)).numpy()
    vt=act.numpy().copy(); vt[pos[:,0].numpy()>=1.0,0]=0.0
np.savetxt("../../../field_compare_shooting.csv",
           np.concatenate([pos.numpy(),vl,vt],1), delimiter=",",
           header="px,py,vlearned_x,vlearned_y,vtrue_x,vtrue_y", comments="")
print("\n4. saved field_compare_shooting.csv")
torch.save(model.state_dict(),"../../../vfield_shooting.pt")
print("saved vfield_shooting.pt")