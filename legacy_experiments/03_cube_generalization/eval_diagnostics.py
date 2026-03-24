#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Sanity checks for variant features:
#   v1: R_of6,  v2: + u_final2,  v3: + R_hf6
# Compares dataset outputs to independent recomputation from memmaps,
# checks rotation orthonormality/det, and u_final2 distribution vs. labels.

import os, json, math
import numpy as np
import torch

# ---- hardcoded paths (edit if yours differ) ----
DATA_ROOT    = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
MEM_TRAIN    = os.path.join(DATA_ROOT, "memmaps_train")
MEM_VAL      = os.path.join(DATA_ROOT, "memmaps_val")
USE_MEM      = MEM_TRAIN        # <- choose which memdir to sanity check
SAMPLE_N     = 20000            # keep it moderate to avoid RAM blowups

# ---- import your dataset helpers ----
from dataset import (
    FinalCornersHandDataset,
    quat_wxyz_to_R_batched
)

# ---------- small helpers ----------
def R_from_6d(r6: np.ndarray) -> np.ndarray:
    a1 = r6[..., :3]
    a2 = r6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    a2p = a2 - (np.sum(b1*a2, axis=-1, keepdims=True) * b1)
    b2 = a2p / (np.linalg.norm(a2p, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    # stack columns
    R = np.stack([b1, b2, b3], axis=-1)  # (...,3,3)
    return R.astype(np.float32)

def load_memmaps(mem_dir):
    meta = json.load(open(os.path.join(mem_dir, "meta.json"), "r"))
    N = int(meta["N"])
    mm = {
        "N": N,
        "dims":   np.memmap(meta["dims_file"],    dtype=np.float32, mode="r", shape=(N,3)),
        "tloc":   np.memmap(meta["tloc_file"],    dtype=np.float32, mode="r", shape=(N,3)),
        "rloc6":  np.memmap(meta["rloc6_file"],   dtype=np.float32, mode="r", shape=(N,6)),
        "final7": np.memmap(meta["final7_file"],  dtype=np.float32, mode="r", shape=(N,7)),
        "init7":  np.memmap(meta["init7_file"],   dtype=np.float32, mode="r", shape=(N,7)),
        "label":  np.memmap(meta["label_file"],   dtype=np.float32, mode="r", shape=(N,1)),
    }
    return mm

def pick_indices(N, k):
    k = int(min(k, N))
    rng = np.random.RandomState(42)
    idx = rng.choice(N, size=k, replace=False)
    idx.sort()
    return idx

def final_up_axis(Rf):  # (...,3,3)
    # which object axis aligns most with world +Z? -> argmax |Rf[2, *]|
    return np.argmax(np.abs(Rf[..., 2, :]), axis=-1)  # (...,)

def u_final2_from(mem, idx):
    # compute R_of and t_f, then in-plane normalized offsets u=(ux,uy)
    f7 = mem["final7"][idx]   # (n,7) [t, q]
    o7 = mem["init7"][idx]
    tloc = mem["tloc"][idx]   # (n,3)
    dims = mem["dims"][idx]   # (n,3)

    Rf = quat_wxyz_to_R_batched(f7[:, 3:])  # (n,3,3)
    Ro = quat_wxyz_to_R_batched(o7[:, 3:])
    Rof = np.einsum("bij,bkj->bik", Rf, Ro)  # Rf @ Ro^T ? Careful.
    # We need R_of = R_f^T R_o:
    Rof = np.einsum("bij,bjk->bik", np.transpose(Rf, (0,2,1)), Ro)  # (n,3,3)

    # t in final frame
    t_f = np.einsum("bij,bj->bi", Rof, tloc)  # (n,3)

    up = final_up_axis(Rf)  # (n,)
    hx = np.empty_like(up, dtype=np.float32)
    hy = np.empty_like(up, dtype=np.float32)
    # up==2 -> Z up -> footprint (x,y) => half=(dx/2, dy/2)
    sel = (up == 2); hx[sel] = 0.5*dims[sel,0]; hy[sel] = 0.5*dims[sel,1]
    # up==0 -> X up -> footprint (y,z)
    sel = (up == 0); hx[sel] = 0.5*dims[sel,1]; hy[sel] = 0.5*dims[sel,2]
    # up==1 -> Y up -> footprint (x,z)
    sel = (up == 1); hx[sel] = 0.5*dims[sel,0]; hy[sel] = 0.5*dims[sel,2]

    # the in-plane components in t_f depend on which axis is "up"
    # if Z up -> use t_f[x], t_f[y]; if X up -> use t_f[y], t_f[z]; if Y up -> use t_f[x], t_f[z]
    u = np.zeros((len(idx), 2), dtype=np.float32)
    mask = (up == 2)
    u[mask, 0] = t_f[mask, 0] / (hx[mask] + 1e-12)
    u[mask, 1] = t_f[mask, 1] / (hy[mask] + 1e-12)
    mask = (up == 0)
    u[mask, 0] = t_f[mask, 1] / (hx[mask] + 1e-12)
    u[mask, 1] = t_f[mask, 2] / (hy[mask] + 1e-12)
    mask = (up == 1)
    u[mask, 0] = t_f[mask, 0] / (hx[mask] + 1e-12)
    u[mask, 1] = t_f[mask, 2] / (hy[mask] + 1e-12)

    # clip like dataset
    u = np.clip(u, -1.0, 1.0)
    return u, Rof, Rf, Ro

def rotation_checks(name, R_batch):
    # Orthonormality and det ~ +1
    RtR = np.matmul(np.transpose(R_batch, (0,2,1)), R_batch)  # (n,3,3)
    I = np.eye(3, dtype=np.float32)[None, :, :]
    frob = np.linalg.norm(RtR - I, axis=(1,2))
    dets = np.linalg.det(R_batch)
    print(f"[{name}]  ||R^T R - I||_F: mean={frob.mean():.3e}, max={frob.max():.3e} | det: mean={dets.mean():.6f}, min={dets.min():.6f}, max={dets.max():.6f}")

def compare_slices(msg, a, b):
    diff = np.abs(a - b)
    print(f"{msg}: mean|Δ|={diff.mean():.3e}, max|Δ|={diff.max():.3e}")

def main():
    print(f"== Sanity on {USE_MEM} ==")
    mm = load_memmaps(USE_MEM)
    N = mm["N"]
    idx = pick_indices(N, SAMPLE_N)
    labels = (mm["label"][idx,0] > 0.5).astype(np.int32)

    # Stats for z-scoring (from train memdir)
    ns = json.load(open(os.path.join(MEM_TRAIN, "stats.json"), "r"))

    # ---- Variant 1: + R_of6 ----
    ds1 = FinalCornersHandDataset(USE_MEM, normalization_stats=ns, is_training=False, variant=1)
    aux1 = []
    for i in idx:
        _, a, _ = ds1[i]
        aux1.append(a.numpy())
    aux1 = np.stack(aux1, axis=0)   # (n, 18)
    Rof6_ds = aux1[:, 12:18]        # last 6

    u_dummy, Rof, Rf, Ro = u_final2_from(mm, idx)  # also gives R_of true
    Rof6_true = np.concatenate([Rof[:, :, 0], Rof[:, :, 1]], axis=1)
    compare_slices("R_of6 (dataset vs recompute)", Rof6_ds, Rof6_true)
    rotation_checks("R_of from 6D (dataset)", R_from_6d(Rof6_ds))
    rotation_checks("R_of (true from quats)", Rof)

    # ---- Variant 2: + u_final2 ----
    ds2 = FinalCornersHandDataset(USE_MEM, normalization_stats=ns, is_training=False, variant=2)
    aux2 = []
    for i in idx:
        _, a, _ = ds2[i]
        aux2.append(a.numpy())
    aux2 = np.stack(aux2, axis=0)   # (n, 20)
    u_ds = aux2[:, 18:20]
    u_true, _, _, _ = u_final2_from(mm, idx)
    compare_slices("u_final2 (dataset vs recompute)", u_ds, u_true)
    # distribution & simple correlation with collisions
    mag = np.linalg.norm(u_ds, axis=1)
    br_all = labels.mean()
    br_edge = labels[mag > 0.8].mean() if np.any(mag > 0.8) else float("nan")
    br_center = labels[mag <= 0.4].mean() if np.any(mag <= 0.4) else float("nan")
    print(f"[u_final2] |u|: mean={mag.mean():.3f}, min={mag.min():.3f}, max={mag.max():.3f}")
    print(f"[u_final2] pct(|u|>0.95)={100.0*np.mean(mag>0.95):.2f}% | base={br_all:.3f} | base@edge={br_edge:.3f} | base@center={br_center:.3f}")

    # ---- Variant 3: + R_hf6 ----
    # Rh = Ro @ R_loc ; R_hf = R_f^T @ R_h
    ds3 = FinalCornersHandDataset(USE_MEM, normalization_stats=ns, is_training=False, variant=3)
    aux3 = []
    for i in idx:
        _, a, _ = ds3[i]
        aux3.append(a.numpy())
    aux3 = np.stack(aux3, axis=0)   # (n, 26)
    Rhf6_ds = aux3[:, 20:26]        # last 6

    # recompute Rhf true
    # reconstruct R_loc from rloc6 via 6D->R
    Rloc = R_from_6d(mm["rloc6"][idx])
    Rh = np.einsum("bij,bjk->bik", Ro, Rloc)
    Rhf = np.einsum("bij,bjk->bik", np.transpose(Rf, (0,2,1)), Rh)
    Rhf6_true = np.concatenate([Rhf[:, :, 0], Rhf[:, :, 1]], axis=1)
    compare_slices("R_hf6 (dataset vs recompute)", Rhf6_ds, Rhf6_true)
    rotation_checks("R_hf from 6D (dataset)", R_from_6d(Rhf6_ds))
    rotation_checks("R_hf (true from quats+r_loc6)", Rhf)

    print("✅ Sanity done.")

if __name__ == "__main__":
    main()
