#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, sys
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ========= HARD-CODED PATHS (edit if needed) =========
EXPERIMENT_FILE = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments/new_experiments.jsonl"
TEST_FILE       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/sim.jsonl"
CKPT_PATH       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/training/R_of6/best_roc.pt"
OUT_ROOT        = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments/eval_runs"
DATA_ROOT       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
TEST_MM         = os.path.join(DATA_ROOT, "memmaps_test")
RUN_NAME        = "R_of6"

# Fixed global operating threshold
TAU = 0.48

# ========= MODEL IMPORT =========
from model import FinalCornersAuxModel  # we will set aux_in based on AUX_OPTION
from dataset import VAL_MM as VAL_MEMMAP_DIR
from dataset import quat_wxyz_to_R_batched, r6_to_R_batched, up_axis_idx_from_Rf, corners_world_from_dims_final_batch
from dataset import DATA_ROOT as DS_ROOT

# Match training options
# 0=BASE(12), 1=+R_of6(18), 2=+tloc_f_inplane(2)->14, 3=+R_hf6(18)
AUX_OPTION = 1

# ========= TEE LOGGER =========
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()
    def isatty(self): return False

# ========= GEOM / FEATS (same conventions as your eval) =========
SIGNS_8x3 = np.array([
    [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    [ 1,-1,-1], [ 1,-1, 1], [ 1, 1,-1], [ 1, 1, 1],
], dtype=np.float32)

def quat_wxyz_to_R(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)

def rot6d_from_R(R):
    return np.concatenate([R[:,0], R[:,1]], axis=0).astype(np.float32)

def r6_to_R(r6):
    a1 = r6[0:3].astype(np.float32)
    a2 = r6[3:6].astype(np.float32)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2p= a2 - (b1 @ a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns

def up_axis_idx_from_Rf_single(Rf):
    zcols = np.abs(Rf[2,:])
    return int(np.argmax(zcols))

def _aux_in_from_option(opt: int) -> int:
    if opt == 1: return 18
    if opt == 2: return 14
    if opt == 3: return 18
    return 12

def zscore(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def build_features(trial, stats, aux_option: int = AUX_OPTION):
    # Accept experiment schema keys
    dims = trial.get("dims", trial.get("object_dimensions"))
    if dims is None:
        raise KeyError("dims/object_dimensions missing in trial")
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])

    final_pose = np.asarray(trial.get("final_pose", trial.get("final_object_pose")), np.float32)
    init_pose  = np.asarray(trial.get("init_pose",  trial.get("init_object_pose")),  np.float32)
    grasp_pose = np.asarray(trial.get("grasp_pose"), np.float32)
    if final_pose is None or init_pose is None or grasp_pose is None:
        raise KeyError("final_pose/init_pose/grasp_pose missing in trial")

    # corners(final, world)
    T = final_pose[:3]
    qf = final_pose[3:7]                                  # [w,x,y,z]
    Rf = quat_wxyz_to_R(qf)
    half = 0.5 * np.array([dx, dy, dz], dtype=np.float32)
    corners_local = SIGNS_8x3 * half
    final_world = (Rf @ corners_local.T).T + T[None, :]
    corners_24 = final_world.reshape(-1)

    # z-score with train stats (same keys as before)
    c24_n = zscore(corners_24, np.asarray(stats["corners_mean"], np.float32),
                             np.asarray(stats["corners_std"],  np.float32)).astype(np.float32)

    # Derive base features from poses
    To = init_pose[:3].astype(np.float32)
    qo = init_pose[3:7].astype(np.float32)
    Th = grasp_pose[:3].astype(np.float32)
    qh = grasp_pose[3:7].astype(np.float32)
    Ro = quat_wxyz_to_R(qo)
    Rh = quat_wxyz_to_R(qh)
    R_of = (Rf.T @ Ro).astype(np.float32)
    R_loc = (Ro.T @ Rh).astype(np.float32)
    t_loc = (Ro.T @ (Th - To)).astype(np.float32)

    tloc_n= zscore(t_loc,      np.asarray(stats["tloc_mean"],    np.float32),
                             np.asarray(stats["tloc_std"],     np.float32)).astype(np.float32)
    dims_n= zscore(np.array([dx,dy,dz], np.float32),
                   np.asarray(stats["dims_mean"], np.float32),
                   np.asarray(stats["dims_std"],  np.float32)).astype(np.float32)
    R6 = rot6d_from_R(R_loc)
    base12 = np.concatenate([tloc_n, R6, dims_n], axis=0).astype(np.float32)

    if aux_option == 0:
        return c24_n, base12

    # Extras depend on aux_option (R_of already computed)

    if aux_option == 1:
        extra = rot6d_from_R(R_of)
        return c24_n, np.concatenate([base12, extra], axis=0).astype(np.float32)
    elif aux_option == 2:
        # t_loc in final frame, in-plane normalization by up-axis
        t_f = (R_of @ t_loc).astype(np.float32)
        up_idx = up_axis_idx_from_Rf_single(Rf)
        if up_idx == 2:
            hx, hy = 0.5*dx, 0.5*dy
            ux, uy = t_f[0]/(hx+1e-12), t_f[1]/(hy+1e-12)
        elif up_idx == 0:
            hx, hy = 0.5*dy, 0.5*dz
            ux, uy = t_f[1]/(hx+1e-12), t_f[2]/(hy+1e-12)
        else:  # up_idx == 1
            hx, hy = 0.5*dx, 0.5*dz
            ux, uy = t_f[0]/(hx+1e-12), t_f[2]/(hy+1e-12)
        extra = np.clip(np.array([ux, uy], np.float32), -1.0, 1.0)
        return c24_n, np.concatenate([base12, extra], axis=0).astype(np.float32)
    elif aux_option == 3:
        R_hf = (R_of @ R_loc).astype(np.float32)
        extra = rot6d_from_R(R_hf)
        return c24_n, np.concatenate([base12, extra], axis=0).astype(np.float32)
    else:
        return c24_n, base12


# ========= METRICS / COHORT HELPERS =========
def predict_trials(trials, ckpt_path, device="auto"):
    if device=="auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(AUX_OPTION), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stats = ckpt["normalization_stats"]

    preds, y_true, p_coll, buckets, dims_all, idx_all = [], [], [], [], [], []
    with torch.no_grad():
        for tr in trials:
            c24, aux12 = build_features(tr, stats, aux_option=AUX_OPTION)
            ct = torch.from_numpy(c24).unsqueeze(0).to(device)
            at = torch.from_numpy(aux12).unsqueeze(0).to(device)
            p = torch.sigmoid(model(ct, at)).item()

            y_true.append(int(tr.get("had_collision", 0)))
            p_coll.append(p)
            buckets.append(tr.get("bucket", None))
            dims_all.append(tr["dims"])
            idx_all.append(tr.get("index"))

            preds.append(dict(index=idx_all[-1], bucket=buckets[-1],
                              dims=[float(x) for x in dims_all[-1]],
                              p_collision=float(p), had_collision=bool(y_true[-1])))
    y = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(p_coll, dtype=np.float32)
    return preds, y, p, buckets, dims_all, idx_all


def _dims_key(d, nd=6):
    return (round(float(d[0]), nd), round(float(d[1]), nd), round(float(d[2]), nd))


def eval_matched_val_by_dims(dims_key, ckpt_path, aux_option: int = AUX_OPTION, batch: int = 200_000):
    # load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(aux_option), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    stats = ckpt["normalization_stats"]

    # memmaps
    meta = json.load(open(os.path.join(VAL_MEMMAP_DIR, "meta.json"), "r"))
    N = int(meta["N"])
    mm_dims   = np.memmap(meta["dims_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_final7 = np.memmap(meta["final7_file"], dtype=np.float32, mode="r", shape=(N,7))
    mm_init7  = np.memmap(meta["init7_file"],  dtype=np.float32, mode="r", shape=(N,7))
    mm_tloc3  = np.memmap(meta["tloc_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_rloc6  = np.memmap(meta["rloc6_file"],  dtype=np.float32, mode="r", shape=(N,6))
    mm_y      = np.memmap(meta["label_file"],  dtype=np.float32, mode="r", shape=(N,1))

    # mask by dims
    keymask = np.all(np.round(mm_dims, 6) == np.array(dims_key, np.float32), axis=1)
    idx = np.where(keymask)[0]
    if idx.size == 0:
        print("[matched-val] no rows with dims", dims_key)
        return None

    probs = []
    labels= []
    c_mu = np.asarray(stats["corners_mean"], np.float32)
    c_sd = np.asarray(stats["corners_std"],  np.float32)
    t_mu = np.asarray(stats["tloc_mean"],    np.float32)
    t_sd = np.asarray(stats["tloc_std"],     np.float32)
    d_mu = np.asarray(stats["dims_mean"],    np.float32)
    d_sd = np.asarray(stats["dims_std"],     np.float32)

    for start in range(0, idx.size, batch):
        sel = idx[start:start+batch]
        dims_b  = mm_dims[sel]
        final_b = mm_final7[sel]
        init_b  = mm_init7[sel]
        tloc_b  = mm_tloc3[sel]
        rloc6_b = mm_rloc6[sel]
        y_b     = (mm_y[sel,0] >= 0.5).astype(np.int32)

        # corners 24
        c24 = corners_world_from_dims_final_batch(dims_b, final_b)
        c24 = (c24 - c_mu) / (c_sd + 1e-8)

        # base 12: tloc_z, rloc6 raw, dims_z
        t_z = (tloc_b - t_mu) / (t_sd + 1e-8)
        d_z = (dims_b - d_mu) / (d_sd + 1e-8)
        aux = np.concatenate([t_z, rloc6_b, d_z], axis=1).astype(np.float32)

        # extras
        if aux_option in (1,2,3):
            Rf = quat_wxyz_to_R_batched(final_b[:,3:7])
            Ro = quat_wxyz_to_R_batched(init_b[:,3:7])
            R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)
            if aux_option == 1:
                extra = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1)
            elif aux_option == 2:
                t_f = np.einsum("bij,bj->bi", R_of, tloc_b)
                up_idx = up_axis_idx_from_Rf(Rf)
                ux = np.empty(t_f.shape[0], np.float32); uy = np.empty_like(ux)
                m = (up_idx==2); ux[m]=t_f[m,0]/(0.5*dims_b[m,0]+1e-12); uy[m]=t_f[m,1]/(0.5*dims_b[m,1]+1e-12)
                m = (up_idx==0); ux[m]=t_f[m,1]/(0.5*dims_b[m,1]+1e-12); uy[m]=t_f[m,2]/(0.5*dims_b[m,2]+1e-12)
                m = (up_idx==1); ux[m]=t_f[m,0]/(0.5*dims_b[m,0]+1e-12); uy[m]=t_f[m,2]/(0.5*dims_b[m,2]+1e-12)
                extra = np.clip(np.stack([ux,uy],axis=1), -1, 1)
            else:
                R_loc = r6_to_R_batched(rloc6_b)
                R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
                extra = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1)
            aux = np.concatenate([aux, extra.astype(np.float32)], axis=1)

        with torch.no_grad():
            ct = torch.from_numpy(c24).to(device)
            at = torch.from_numpy(aux).to(device)
            pr = torch.sigmoid(model(ct, at)).view(-1).cpu().numpy()
        probs.append(pr); labels.append(y_b)

    p = np.concatenate(probs); y = np.concatenate(labels)
    base = float(y.mean());
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    # simple best tau sweep
    taus = np.linspace(0,1,201)
    best_acc = -1.0; best_tau = 0.5
    for t in taus:
        yh = (p>=t).astype(np.int32)
        tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
        acc = (tp+tn)/max(1,len(y))
        if acc>best_acc: best_acc, best_tau = acc, float(t)
    acc_tau = ((p>=TAU).astype(np.int32)==y).mean()
    print(f"\n[matched-val] dims={dims_key}  n={len(y)}  base={base:.3f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f}={acc_tau:.4f}  best_τ={best_tau:.2f} acc={best_acc:.4f}")
    return dict(n=len(y), base=base, auc_roc=auc_roc, auc_pr=auc_pr, acc_tau=float(acc_tau), best_tau=best_tau, best_acc=best_acc)


def eval_approx_val_by_dims(target_dims, ckpt_path, aux_option: int = AUX_OPTION,
                            eps_list=(1e-4,5e-4,1e-3,5e-3,1e-2), topk=200_000, chunk=2_000_000, batch=200_000):
    target = np.array(target_dims, np.float32)
    # load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(aux_option), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    stats = ckpt["normalization_stats"]

    # memmaps
    meta = json.load(open(os.path.join(VAL_MEMMAP_DIR, "meta.json"), "r"))
    N = int(meta["N"])
    mm_dims   = np.memmap(meta["dims_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_final7 = np.memmap(meta["final7_file"], dtype=np.float32, mode="r", shape=(N,7))
    mm_init7  = np.memmap(meta["init7_file"],  dtype=np.float32, mode="r", shape=(N,7))
    mm_tloc3  = np.memmap(meta["tloc_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_rloc6  = np.memmap(meta["rloc6_file"],  dtype=np.float32, mode="r", shape=(N,6))
    mm_y      = np.memmap(meta["label_file"],  dtype=np.float32, mode="r", shape=(N,1))

    # epsilon ladder
    idx = None
    for eps in eps_list:
        m = (np.abs(mm_dims[:,0]-target[0])<=eps) & (np.abs(mm_dims[:,1]-target[1])<=eps) & (np.abs(mm_dims[:,2]-target[2])<=eps)
        if m.any():
            idx = np.where(m)[0]
            print(f"[approx-match] dims≈{tuple(target_dims)} eps={eps} -> n={idx.size}")
            break
    # KNN fallback
    if idx is None:
        keep_idx = []
        keep_d2  = []
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            d = mm_dims[s:e]
            diff = d - target[None,:]
            dist2 = (diff*diff).sum(axis=1)
            k = min(topk, dist2.size)
            sel = np.argpartition(dist2, k-1)[:k]
            keep_idx.append(sel + s)
            keep_d2.append(dist2[sel])
        kk = np.concatenate(keep_idx); kd = np.concatenate(keep_d2)
        K = min(topk, kd.size)
        selg = np.argpartition(kd, K-1)[:K]
        idx = kk[selg]
        print(f"[approx-match] KNN topK={K} for dims≈{tuple(target_dims)}")

    # score selected idx (reuse code from exact evaluator)
    probs = []
    labels= []
    c_mu = np.asarray(stats["corners_mean"], np.float32)
    c_sd = np.asarray(stats["corners_std"],  np.float32)
    t_mu = np.asarray(stats["tloc_mean"],    np.float32)
    t_sd = np.asarray(stats["tloc_std"],     np.float32)
    d_mu = np.asarray(stats["dims_mean"],    np.float32)
    d_sd = np.asarray(stats["dims_std"],     np.float32)

    for start in range(0, idx.size, batch):
        sel = idx[start:start+batch]
        dims_b  = mm_dims[sel]
        final_b = mm_final7[sel]
        init_b  = mm_init7[sel]
        tloc_b  = mm_tloc3[sel]
        rloc6_b = mm_rloc6[sel]
        y_b     = (mm_y[sel,0] >= 0.5).astype(np.int32)

        c24 = corners_world_from_dims_final_batch(dims_b, final_b)
        c24 = (c24 - c_mu) / (c_sd + 1e-8)
        t_z = (tloc_b - t_mu) / (t_sd + 1e-8)
        d_z = (dims_b - d_mu) / (d_sd + 1e-8)
        aux = np.concatenate([t_z, rloc6_b, d_z], axis=1).astype(np.float32)

        if aux_option in (1,2,3):
            Rf = quat_wxyz_to_R_batched(final_b[:,3:7])
            Ro = quat_wxyz_to_R_batched(init_b[:,3:7])
            R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)
            if aux_option == 1:
                extra = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1)
            elif aux_option == 2:
                t_f = np.einsum("bij,bj->bi", R_of, tloc_b)
                up_idx = up_axis_idx_from_Rf(Rf)
                ux = np.empty(t_f.shape[0], np.float32); uy = np.empty_like(ux)
                m = (up_idx==2); ux[m]=t_f[m,0]/(0.5*dims_b[m,0]+1e-12); uy[m]=t_f[m,1]/(0.5*dims_b[m,1]+1e-12)
                m = (up_idx==0); ux[m]=t_f[m,1]/(0.5*dims_b[m,1]+1e-12); uy[m]=t_f[m,2]/(0.5*dims_b[m,2]+1e-12)
                m = (up_idx==1); ux[m]=t_f[m,0]/(0.5*dims_b[m,0]+1e-12); uy[m]=t_f[m,2]/(0.5*dims_b[m,2]+1e-12)
                extra = np.clip(np.stack([ux,uy],axis=1), -1, 1)
            else:
                R_loc = r6_to_R_batched(rloc6_b)
                R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
                extra = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1)
            aux = np.concatenate([aux, extra.astype(np.float32)], axis=1)

        with torch.no_grad():
            ct = torch.from_numpy(c24).to(device)
            at = torch.from_numpy(aux).to(device)
            pr = torch.sigmoid(model(ct, at)).view(-1).cpu().numpy()
        probs.append(pr); labels.append(y_b)

    p = np.concatenate(probs); y = np.concatenate(labels)
    base = float(y.mean());
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    taus = np.linspace(0,1,201)
    best_acc = -1.0; best_tau = 0.5
    for t in taus:
        yh = (p>=t).astype(np.int32)
        tp = int(((yh==1)&(y==1)).sum()); tn = int(((yh==0)&(y==0)).sum())
        acc = (tp+tn)/max(1,len(y))
        if acc>best_acc: best_acc, best_tau = acc, float(t)
    acc_tau = ((p>=TAU).astype(np.int32)==y).mean()
    print(f"\n[approx-val] dims≈{tuple(target_dims)}  n={len(y)}  base={base:.3f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f}={acc_tau:.4f}  best_τ={best_tau:.2f} acc={best_acc:.4f}")
    return dict(n=len(y), base=base, auc_roc=auc_roc, auc_pr=auc_pr, acc_tau=float(acc_tau), best_tau=best_tau, best_acc=best_acc)


# ========= TEST.JSON EVALUATION (streamed) =========
def _stream_array(path: str):
    with open(path, "r") as f:
        in_arr = False; in_str = False; esc = False; depth = 0; buf = []
        while True:
            ch = f.read(1)
            if not ch: break
            if in_str:
                buf.append(ch)
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True; buf.append(ch); continue
            if ch == '[': in_arr = True; continue
            if not in_arr: continue
            if ch == '{':
                buf = ['{']; depth = 1
                while True:
                    c = f.read(1)
                    if not c: break
                    buf.append(c)
                    if c == '"': in_str = not in_str
                    elif not in_str:
                        if c == '{': depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                yield json.loads(''.join(buf))
                                break
                continue
            if ch == ']': return


def eval_test_json(test_json_path: str = None, ckpt_path: str = CKPT_PATH, aux_option: int = AUX_OPTION, batch: int = 200_000):
    if test_json_path is None:
        test_json_path = os.path.join(DS_ROOT, "combined_data", "test.json")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(aux_option), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    stats = ckpt["normalization_stats"]

    c_mu = np.asarray(stats["corners_mean"], np.float32)
    c_sd = np.asarray(stats["corners_std"],  np.float32)
    t_mu = np.asarray(stats["tloc_mean"],    np.float32)
    t_sd = np.asarray(stats["tloc_std"],     np.float32)
    d_mu = np.asarray(stats["dims_mean"],    np.float32)
    d_sd = np.asarray(stats["dims_std"],     np.float32)

    probs = []; labels = []
    dims_b=[]; final_b=[]; init_b=[]; tloc_b=[]; rloc6_b=[]; y_b=[]

    def _flush():
        nonlocal probs, labels, dims_b, final_b, init_b, tloc_b, rloc6_b, y_b
        if not dims_b: return
        dims = np.asarray(dims_b, np.float32)
        fin7 = np.asarray(final_b, np.float32)
        ini7 = np.asarray(init_b,  np.float32)
        tl3  = np.asarray(tloc_b,  np.float32)
        rl6  = np.asarray(rloc6_b, np.float32)
        yy   = np.asarray(y_b,     np.int32)

        c24 = corners_world_from_dims_final_batch(dims, fin7)
        c24 = (c24 - c_mu) / (c_sd + 1e-8)
        t_z = (tl3  - t_mu) / (t_sd + 1e-8)
        d_z = (dims - d_mu) / (d_sd + 1e-8)
        aux = np.concatenate([t_z, rl6, d_z], axis=1).astype(np.float32)

        if aux_option in (1,2,3):
            # copy to avoid in-place normalization on read-only memmaps
            Rf = quat_wxyz_to_R_batched(fin7[:,3:7].copy())
            Ro = quat_wxyz_to_R_batched(ini7[:,3:7].copy())
            R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)
            if aux_option == 1:
                extra = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1)
            elif aux_option == 2:
                t_f = np.einsum("bij,bj->bi", R_of, tl3)
                up_idx = up_axis_idx_from_Rf(Rf)
                B = t_f.shape[0]
                ux = np.empty(B, np.float32); uy = np.empty(B, np.float32)
                m = (up_idx==2); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12)
                m = (up_idx==0); ux[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                m = (up_idx==1); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                extra = np.clip(np.stack([ux,uy],axis=1), -1, 1)
            else:
                R_loc = r6_to_R_batched(rl6)
                R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
                extra = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1)
            aux = np.concatenate([aux, extra.astype(np.float32)], axis=1)

        with torch.no_grad():
            ct = torch.from_numpy(c24).to(device)
            at = torch.from_numpy(aux).to(device)
            pr = torch.sigmoid(model(ct, at)).view(-1).cpu().numpy()
        probs.append(pr); labels.append(yy)

        dims_b.clear(); final_b.clear(); init_b.clear(); tloc_b.clear(); rloc6_b.clear(); y_b.clear()

    # first pass: count rows for progress bar
    n_total = 0
    for _ in _stream_array(test_json_path):
        n_total += 1

    n = 0
    pbar = tqdm(total=n_total, desc="Scoring test.json", unit="obj", dynamic_ncols=True)
    for obj in _stream_array(test_json_path):
        dims_b.append(obj["object_dimensions"])
        final_b.append(obj["final_object_pose"])
        init_b.append(obj["init_object_pose"])
        tloc_b.append(obj["t_loc"])
        rloc6_b.append(obj["R_loc6"])
        y_b.append(1 if float(obj["collision_label"])>0.5 else 0)
        n += 1
        pbar.update(1)
        if len(dims_b) >= batch:
            _flush()
    _flush()
    pbar.close()

    p = np.concatenate(probs); y = np.concatenate(labels)
    base = float(y.mean())
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    acc_tau = ((p>=TAU).astype(np.int32)==y).mean()
    print(f"\n[test.json] n={len(y)}  base={base:.3f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f}={acc_tau:.4f}")


def eval_test_memmaps(memmap_dir: str, ckpt_path: str = CKPT_PATH, aux_option: int = AUX_OPTION, batch: int = 400_000):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(aux_option), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    stats = ckpt["normalization_stats"]

    c_mu = np.asarray(stats["corners_mean"], np.float32)
    c_sd = np.asarray(stats["corners_std"],  np.float32)
    t_mu = np.asarray(stats["tloc_mean"],    np.float32)
    t_sd = np.asarray(stats["tloc_std"],     np.float32)
    d_mu = np.asarray(stats["dims_mean"],    np.float32)
    d_sd = np.asarray(stats["dims_std"],     np.float32)

    meta = json.load(open(os.path.join(memmap_dir, "meta.json"), "r"))
    N = int(meta["N"])
    mm_dims   = np.memmap(meta["dims_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_final7 = np.memmap(meta["final7_file"], dtype=np.float32, mode="r", shape=(N,7))
    mm_init7  = np.memmap(meta["init7_file"],  dtype=np.float32, mode="r", shape=(N,7))
    mm_tloc3  = np.memmap(meta["tloc_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_rloc6  = np.memmap(meta["rloc6_file"],  dtype=np.float32, mode="r", shape=(N,6))
    mm_y      = np.memmap(meta["label_file"],  dtype=np.float32, mode="r", shape=(N,1))

    probs = []; labels = []
    pbar = tqdm(total=N, desc="Scoring test memmaps", unit="obj", dynamic_ncols=True)
    for s in range(0, N, batch):
        e = min(N, s+batch); B = e - s
        dims = mm_dims[s:e]
        fin7 = mm_final7[s:e]
        ini7 = mm_init7[s:e]
        tl3  = mm_tloc3[s:e]
        rl6  = mm_rloc6[s:e]
        yy   = (mm_y[s:e,0] >= 0.5).astype(np.int32)

        c24 = corners_world_from_dims_final_batch(dims, fin7)
        c24 = (c24 - c_mu) / (c_sd + 1e-8)
        t_z = (tl3  - t_mu) / (t_sd + 1e-8)
        d_z = (dims - d_mu) / (d_sd + 1e-8)
        aux = np.concatenate([t_z, rl6, d_z], axis=1).astype(np.float32)

        if aux_option in (1,2,3):
            # copy to avoid in-place normalization on read-only memmaps
            Rf = quat_wxyz_to_R_batched(fin7[:,3:7].copy())
            Ro = quat_wxyz_to_R_batched(ini7[:,3:7].copy())
            R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)
            if aux_option == 1:
                extra = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1)
            elif aux_option == 2:
                t_f = np.einsum("bij,bj->bi", R_of, tl3)
                up_idx = up_axis_idx_from_Rf(Rf)
                ux = np.empty(B, np.float32); uy = np.empty(B, np.float32)
                m = (up_idx==2); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12)
                m = (up_idx==0); ux[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                m = (up_idx==1); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                extra = np.clip(np.stack([ux,uy],axis=1), -1, 1)
            else:
                R_loc = r6_to_R_batched(rl6)
                R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
                extra = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1)
            aux = np.concatenate([aux, extra.astype(np.float32)], axis=1)

        with torch.no_grad():
            ct = torch.from_numpy(c24).to(device)
            at = torch.from_numpy(aux).to(device)
            pr = torch.sigmoid(model(ct, at)).view(-1).cpu().numpy()
        probs.append(pr); labels.append(yy)
        pbar.update(B)
    pbar.close()

    p = np.concatenate(probs); y = np.concatenate(labels)
    base = float(y.mean())
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    acc_tau = ((p>=TAU).astype(np.int32)==y).mean()
    print(f"\n[test.memmaps] n={len(y)}  base={base:.3f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f}={acc_tau:.4f}")



def eval_experiment_jsonl(experiment_file: str = EXPERIMENT_FILE, ckpt_path: str = CKPT_PATH, aux_option: int = AUX_OPTION, batch: int = 400_000):
    """Evaluate a JSONL experiment file with the same batched pipeline as memmaps.

    Each line should contain at least: dims or object_dimensions; final_pose or final_object_pose;
    init_pose or init_object_pose; grasp_pose; and optionally had_collision or collision_label.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCornersAuxModel(aux_in=_aux_in_from_option(aux_option), use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    stats = ckpt["normalization_stats"]

    c_mu = np.asarray(stats["corners_mean"], np.float32)
    c_sd = np.asarray(stats["corners_std"],  np.float32)
    t_mu = np.asarray(stats["tloc_mean"],    np.float32)
    t_sd = np.asarray(stats["tloc_std"],     np.float32)
    d_mu = np.asarray(stats["dims_mean"],    np.float32)
    d_sd = np.asarray(stats["dims_std"],     np.float32)

    # count lines for progress bar
    total = 0
    with open(experiment_file, "r") as f:
        for line in f:
            if line.strip():
                total += 1

    probs = []
    labels = []
    buckets = []

    def _label_from(obj):
        if "had_collision" in obj:
            return 1 if int(obj["had_collision"]) == 1 else 0
        if "collision_label" in obj:
            return 1 if float(obj["collision_label"]) > 0.5 else 0
        return 0

    dims_b = []; final_b = []; init_b = []; grasp_b = []; y_b = []; buck_b = []
    pbar = tqdm(total=total, desc="Scoring experiment.jsonl", unit="obj", dynamic_ncols=True)

    def _flush():
        nonlocal probs, labels, buckets, dims_b, final_b, init_b, grasp_b, y_b, buck_b
        if not dims_b:
            return
        dims = np.asarray(dims_b,  np.float32)
        fin7 = np.asarray(final_b, np.float32)
        ini7 = np.asarray(init_b,  np.float32)
        grp7 = np.asarray(grasp_b, np.float32)
        yy   = np.asarray(y_b,     np.int32)

        # corners 24 (final frame world)
        c24 = corners_world_from_dims_final_batch(dims, fin7)
        c24 = (c24 - c_mu) / (c_sd + 1e-8)

        # rotations/locals
        Rf = quat_wxyz_to_R_batched(fin7[:,3:7].copy())
        Ro = quat_wxyz_to_R_batched(ini7[:,3:7].copy())
        Rh = quat_wxyz_to_R_batched(grp7[:,3:7].copy())

        # t_loc = Ro^T (Th - To)
        To = ini7[:,0:3].astype(np.float32)
        Th = grp7[:,0:3].astype(np.float32)
        dT = Th - To
        tloc = np.einsum("bij,bj->bi", np.transpose(Ro,(0,2,1)), dT)
        R_loc = np.einsum("bij,bjk->bik", np.transpose(Ro,(0,2,1)), Rh)

        t_z = (tloc - t_mu) / (t_sd + 1e-8)
        d_z = (dims - d_mu) / (d_sd + 1e-8)
        rloc6 = np.concatenate([R_loc[:,:,0], R_loc[:,:,1]], axis=1).astype(np.float32)
        aux = np.concatenate([t_z, rloc6, d_z], axis=1).astype(np.float32)

        if aux_option in (1,2,3):
            R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)
            if aux_option == 1:
                extra = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1)
            elif aux_option == 2:
                t_f = np.einsum("bij,bj->bi", R_of, tloc)
                up_idx = up_axis_idx_from_Rf(Rf)
                B = t_f.shape[0]
                ux = np.empty(B, np.float32); uy = np.empty(B, np.float32)
                m = (up_idx==2); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12)
                m = (up_idx==0); ux[m]=t_f[m,1]/(0.5*dims[m,1]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                m = (up_idx==1); ux[m]=t_f[m,0]/(0.5*dims[m,0]+1e-12); uy[m]=t_f[m,2]/(0.5*dims[m,2]+1e-12)
                extra = np.clip(np.stack([ux,uy],axis=1), -1, 1)
            else:
                R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
                extra = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1)
            aux = np.concatenate([aux, extra.astype(np.float32)], axis=1)

        with torch.no_grad():
            ct = torch.from_numpy(c24).to(device)
            at = torch.from_numpy(aux).to(device)
            pr = torch.sigmoid(model(ct, at)).view(-1).cpu().numpy()
        probs.append(pr); labels.append(yy); buckets.extend(buck_b)

        dims_b.clear(); final_b.clear(); init_b.clear(); grasp_b.clear(); y_b.clear(); buck_b.clear()

    with open(experiment_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            d = obj.get("dims", obj.get("object_dimensions"))
            fin = obj.get("final_pose", obj.get("final_object_pose"))
            ini = obj.get("init_pose",  obj.get("init_object_pose"))
            grp = obj.get("grasp_pose", None)
            if d is None or fin is None or ini is None or grp is None:
                pbar.update(1)
                continue
            dims_b.append(d)
            final_b.append(fin)
            init_b.append(ini)
            grasp_b.append(grp)
            y_b.append(_label_from(obj))
            buck_b.append(obj.get("bucket", None))
            if len(dims_b) >= batch:
                _flush()
            pbar.update(1)
    _flush()
    pbar.close()

    p = np.concatenate(probs) if len(probs)>0 else np.zeros((0,), np.float32)
    y = np.concatenate(labels) if len(labels)>0 else np.zeros((0,), np.int32)
    if y.size == 0:
        print("[experiment.jsonl] no valid rows")
        return dict(n=0)
    base = float(y.mean())
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    acc_tau = ((p>=TAU).astype(np.int32)==y).mean()
    print(f"\n[experiment.jsonl] n={len(y)}  base={base:.3f}  AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f}={acc_tau:.4f}")
    # optional per-bucket summary if buckets present
    if any(b is not None for b in buckets):
        try:
            pb = per_bucket_auc(y, p, np.array([b if b is not None else "" for b in buckets]))
            if pb:
                print("\n[experiment.jsonl] per-bucket AUCs:")
                for b in ("ADJACENT","OPPOSITE","MEDIUM","SMALL"):
                    if b in pb:
                        r = pb[b]
                        print(f"{b:<9}  n={r['n']:>5}  base={r['base']:.3f}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}")
        except Exception:
            pass
    return dict(n=len(y), base=base, auc_roc=auc_roc, auc_pr=auc_pr, acc_tau=float(acc_tau))



def compare_results_vs_sim_by_index(sim_jsonl_path: str, experiment_results_path: str):
    """Compare collision labels between sim.jsonl (collision_label) and experiment_results.jsonl (had_collision),
    aligned by index field from experiment results (index is 0-based line number in sim.jsonl).

    Prints counts of matches/mismatches and FP/FN.
    """
    # Load sim subset as list to preserve ordering
    sim_rows = []
    with open(sim_jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sim_rows.append(obj)
            except Exception:
                continue
    N = len(sim_rows)
    if N == 0:
        print(f"[compare] No rows loaded from sim file: {sim_jsonl_path}")
        return

    # Helper to get sim collision label
    def sim_label(idx: int) -> int:
        if idx < 0 or idx >= N:
            return -1  # invalid
        row = sim_rows[idx]
        if "collision_label" in row:
            try:
                return 1 if float(row["collision_label"]) > 0.5 else 0
            except Exception:
                return 0
        # If missing, default to 0 (no collision)
        return 0

    total = 0
    matches = 0
    mismatches = 0
    fp = 0  # had_collision==1 but sim_label==0
    fn = 0  # had_collision==0 but sim_label==1
    missing = 0  # indices out of range

    # Optional per-bucket mismatch counts
    per_bucket = {}

    with open(experiment_results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            # Filter: only keep outcomes we care about
            outcome_val = r.get("outcome", "")
            if outcome_val not in ("collision_limit", "success"):
                continue
            if "index" not in r:
                continue
            idx = int(r["index"])
            if idx < 0 or idx >= N:
                missing += 1
                continue
            y_sim = sim_label(idx)
            if y_sim < 0:
                missing += 1
                continue
            y_exp = 1 if bool(r.get("had_collision", False)) else 0
            total += 1
            if y_sim == y_exp:
                matches += 1
            else:
                mismatches += 1
                if y_exp == 1 and y_sim == 0:
                    fp += 1
                elif y_exp == 0 and y_sim == 1:
                    fn += 1
                b = r.get("bucket", "UNKNOWN")
                per_bucket[b] = per_bucket.get(b, 0) + 1

    if total == 0:
        print("[compare] No comparable rows (check paths and formats)")
        return

    acc = matches / total
    print("\n================ Collision label agreement (results vs sim) ================")
    print(f"sim file: {sim_jsonl_path}")
    print(f"results : {experiment_results_path}")
    print(f"loaded sim rows: {N}")
    print(f"compared: {total}  matches: {matches}  mismatches: {mismatches}  acc={acc:.4f}  missing_idx={missing}")
    print(f"  FP (results had_collision=1, sim_label=0): {fp}")
    print(f"  FN (results had_collision=0, sim_label=1): {fn}")
    if per_bucket:
        print("mismatches per bucket:")
        for k,v in sorted(per_bucket.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")



def confusion_masks(y, p, tau):
    yhat = (p >= tau).astype(np.int32)
    FP = (yhat==1) & (y==0)
    FN = (yhat==0) & (y==1)
    TP = (yhat==1) & (y==1)
    TN = (yhat==0) & (y==0)
    return FP, FN, TP, TN, yhat

def per_bucket_auc(y, p, buckets):
    out = {}
    for b in ("ADJACENT","OPPOSITE","MEDIUM","SMALL"):
        m = (np.asarray(buckets)==b)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        out[b] = dict(
            roc_auc=float(roc_auc_score(y[m], p[m])),
            pr_auc=float(average_precision_score(y[m], p[m])),
            n=int(m.sum()),
            base=float(y[m].mean()))
    return out

def reliability(y, p, bins=12):
    edges = np.linspace(0,1,bins+1)
    idx = np.digitize(p, edges, right=True)
    rows = []
    for b in range(1, bins+1):
        m = (idx==b)
        if m.sum()==0: continue
        rows.append(dict(bin=b, n=int(m.sum()),
                         prob=float(p[m].mean()),
                         freq=float(y[m].mean())))
    return rows

def aspect_ratio(dims):
    d = np.asarray(dims, np.float32)
    return float(d.min() / (d.max() + 1e-12))

def norm_dims(dims):
    d = np.asarray(dims, np.float32)
    M = d.max() + 1e-12
    return list((d / M).tolist())  # each in (0,1], max->1

def binned_stats(values, mask, labels, bins):
    """values: array; mask selects cohort; labels=y or error mask."""
    v = np.asarray(values, np.float32)
    m = np.asarray(mask, bool)
    y = np.asarray(labels)
    edges = np.asarray(bins, np.float32)
    out = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        sel = m & (v >= lo) & (v < hi)
        n = int(sel.sum())
        if n==0:
            out.append(dict(lo=float(lo), hi=float(hi), n=0, rate=float("nan")))
        else:
            out.append(dict(lo=float(lo), hi=float(hi), n=n, rate=float(y[sel].mean())))
    return out

# ========= PLOTTING (compact) =========
def plot_adjacent_aspect_fpfn(ar, FP, FN, TP, TN, out_dir, tau):
    def hist_pair(mask_pos, mask_neg, title, fname):
        plt.figure(figsize=(6,4))
        plt.hist(ar[mask_pos], bins=30, alpha=0.6, label="errors", density=True)
        plt.hist(ar[mask_neg], bins=30, alpha=0.6, label="correct", density=True)
        plt.xlabel("aspect = min(d)/max(d)"); plt.ylabel("density")
        plt.title(title + f"  (τ={tau:.2f})"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=200); plt.close()

    hist_pair(FP, TN, "ADJACENT: FP vs TN by aspect", "adjacent_fp_tn_aspect.png")
    hist_pair(FN, TP, "ADJACENT: FN vs TP by aspect", "adjacent_fn_tp_aspect.png")

def plot_bucket_reliability(y, p, buckets, out_dir):
    for b in ("ADJACENT","OPPOSITE","MEDIUM","SMALL"):
        m = (np.asarray(buckets)==b)
        if m.sum()==0: continue
        rows = reliability(y[m], p[m], bins=12)
        if not rows: continue
        xs = [r["prob"] for r in rows]
        ys = [r["freq"] for r in rows]
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1],"--", lw=1)
        plt.plot(xs, ys, marker="o")
        sizes = [4+0.6*r["n"]**0.5 for r in rows]
        for (x,yv,s) in zip(xs, ys, sizes):
            plt.scatter([x],[yv], s=s)
        plt.xlabel("predicted prob"); plt.ylabel("empirical freq")
        plt.title(f"Reliability: {b}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"reliability_{b.lower()}.png"), dpi=200); plt.close()

def plot_bucket_roc_pr(y, p, buckets, out_dir):
    for b in ("ADJACENT","OPPOSITE","MEDIUM","SMALL"):
        m = (np.asarray(buckets)==b)
        if m.sum()<2 or len(np.unique(y[m]))<2: continue
        fpr, tpr, _ = roc_curve(y[m], p[m])
        prec, rec, _ = precision_recall_curve(y[m], p[m])
        plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {b}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"roc_{b.lower()}.png"), dpi=200); plt.close()
        plt.figure(figsize=(5,4)); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {b}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"pr_{b.lower()}.png"), dpi=200); plt.close()

# ========= MAIN =========
def main():
    out_dir = os.path.join(OUT_ROOT, RUN_NAME)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "diag.log"), "a")
    sys.stdout = Tee(sys.stdout, log_file)

    # Load trials
    trials = []
    with open(EXPERIMENT_FILE, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            trials.append(json.loads(line))

    print(f"Loaded {len(trials)} trials")
    print(f"Checkpoint: {CKPT_PATH}")

    # Predict
    preds, y, p, buckets, dims_all, idx_all = predict_trials(trials, CKPT_PATH, device="auto")

    # Global metrics
    auc_roc = float(roc_auc_score(y, p)) if len(np.unique(y))>1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(np.unique(y))>1 else float("nan")
    yhat = (p >= TAU).astype(np.int32)
    TP = int(((yhat==1)&(y==1)).sum())
    FP = int(((yhat==1)&(y==0)).sum())
    FN = int(((yhat==0)&(y==1)).sum())
    TN = int(((yhat==0)&(y==0)).sum())
    acc = (TP+TN)/max(1,len(y))

    print("\n======================== Global ========================")
    print(f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Acc@τ={TAU:.2f} = {acc:.4f}")
    print(f"TP={TP} TN={TN} FP={FP} FN={FN}  base_rate={y.mean():.3f}")

    # Per-bucket ROC/PR summary
    pb_auc = per_bucket_auc(y, p, buckets)
    print("\n---------------- Per-bucket AUCs ----------------")
    for b in ("ADJACENT","OPPOSITE","MEDIUM","SMALL"):
        if b in pb_auc:
            r = pb_auc[b]
            print(f"{b:<9}  n={r['n']:>5}  base={r['base']:.3f}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}")

    # Confusions & margins (p - τ)
    FP_m, FN_m, TP_m, TN_m, _ = confusion_masks(y, p, TAU)
    margin = p - TAU

    # Aspect ratio & normalized dims for ALL + ADJACENT
    ar = np.array([aspect_ratio(d) for d in dims_all], dtype=np.float32)
    nd = np.array([norm_dims(d) for d in dims_all], dtype=np.float32)  # [N,3], each row max=1

    adj_mask = (np.asarray(buckets)=="ADJACENT")
    opp_mask = (np.asarray(buckets)=="OPPOSITE")
    med_mask = (np.asarray(buckets)=="MEDIUM")
    sml_mask = (np.asarray(buckets)=="SMALL")

    # Print margin stats per cohort (helps see overlap)
    def q(a): 
        a=np.asarray(a, np.float32)
        return np.percentile(a, [5,25,50,75,95]).round(4).tolist()

    print("\n---------------- ADJACENT: margins (p-τ) ----------------")
    print(f"FP count={int((FP_m & adj_mask).sum()):>5}  margins q=[5,25,50,75,95]: {q(margin[FP_m & adj_mask]) if (FP_m & adj_mask).any() else 'NA'}")
    print(f"TN count={int((TN_m & adj_mask).sum()):>5}  margins q=[5,25,50,75,95]: {q(margin[TN_m & adj_mask]) if (TN_m & adj_mask).any() else 'NA'}")
    print(f"FN count={int((FN_m & adj_mask).sum()):>5}  margins q=[5,25,50,75,95]: {q(margin[FN_m & adj_mask]) if (FN_m & adj_mask).any() else 'NA'}")
    print(f"TP count={int((TP_m & adj_mask).sum()):>5}  margins q=[5,25,50,75,95]: {q(margin[TP_m & adj_mask]) if (TP_m & adj_mask).any() else 'NA'}")

    # Error rate vs aspect bins per bucket
    ar_bins = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.01]
    print("\n---------------- Error rate vs aspect(min/max) bins ----------------")
    for name, m in [("ADJACENT", adj_mask), ("OPPOSITE", opp_mask), ("MEDIUM", med_mask), ("SMALL", sml_mask)]:
        if not m.any(): continue
        # FP rate among predicted positives; FN rate among positives (y==1)
        fp_rate_bins = binned_stats(ar, m & (yhat==1), (y==0), ar_bins)
        fn_rate_bins = binned_stats(ar, m & (y==1),  (yhat==0), ar_bins)
        print(f"\n{name}:")
        print("  FP-rate among predicted-pos (by aspect bins):")
        for r in fp_rate_bins:
            print(f"    [{r['lo']:.2f},{r['hi']:.2f})  n={r['n']:>4}  fp_rate={r['rate']:.3f}")
        print("  FN-rate among true-pos (by aspect bins):")
        for r in fn_rate_bins:
            print(f"    [{r['lo']:.2f},{r['hi']:.2f})  n={r['n']:>4}  fn_rate={r['rate']:.3f}")

    # Save CSVs for all ADJACENT errors (full, not top-k)
    import csv
    with open(os.path.join(out_dir, "adjacent_fp.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index","dx","dy","dz","aspect","p","margin"])
        for i in np.where(FP_m & adj_mask)[0]:
            d = preds[i]["dims"]; w.writerow([preds[i]["index"], d[0], d[1], d[2], aspect_ratio(d), float(p[i]), float(margin[i])])
    with open(os.path.join(out_dir, "adjacent_fn.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index","dx","dy","dz","aspect","p","margin"])
        for i in np.where(FN_m & adj_mask)[0]:
            d = preds[i]["dims"]; w.writerow([preds[i]["index"], d[0], d[1], d[2], aspect_ratio(d), float(p[i]), float(margin[i])])

    # Plots: ADJACENT error vs correct by aspect; reliability & ROC/PR per bucket
    plot_adjacent_aspect_fpfn(ar[adj_mask], FP_m[adj_mask], FN_m[adj_mask], TP_m[adj_mask], TN_m[adj_mask], out_dir, TAU)
    plot_bucket_reliability(y, p, buckets, out_dir)
    plot_bucket_roc_pr(y, p, buckets, out_dir)

    # Save a compact JSON summary with key diagnostics
    summary = {
        "tau": float(TAU),
        "global": {  # OK as a string key
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "acc": float(acc),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN
        },
        "per_bucket_auc": pb_auc,
        "adjacent": {
            "counts": {
                "FP": int((FP_m & adj_mask).sum()),
                "FN": int((FN_m & adj_mask).sum()),
                "TP": int((TP_m & adj_mask).sum()),
                "TN": int((TN_m & adj_mask).sum())
            },
            "margin_quantiles": {
                "FP": q(margin[FP_m & adj_mask]) if (FP_m & adj_mask).any() else None,
                "TN": q(margin[TN_m & adj_mask]) if (TN_m & adj_mask).any() else None,
                "FN": q(margin[FN_m & adj_mask]) if (FN_m & adj_mask).any() else None,
                "TP": q(margin[TP_m & adj_mask]) if (TP_m & adj_mask).any() else None
            },
            "aspect_bins": ar_bins
        }
    }

    with open(os.path.join(out_dir, "diagnostics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Saved:")
    print(f"  • logs:                {os.path.join(out_dir, 'diag.log')}")
    print(f"  • adj FP/FN CSVs:      {os.path.join(out_dir, 'adjacent_fp.csv')} / adjacent_fn.csv")
    print(f"  • per-bucket plots:    reliability_*.png, roc_*.png, pr_*.png")
    print(f"  • ADJ aspect plots:    adjacent_fp_tn_aspect.png, adjacent_fn_tp_aspect.png")
    print(f"  • summary JSON:        {os.path.join(out_dir, 'diagnostics_summary.json')}")

if __name__ == "__main__":
    compare_results_vs_sim_by_index(TEST_FILE, EXPERIMENT_FILE)
    # main()
    # eval_test_memmaps(TEST_MM, CKPT_PATH, aux_option=AUX_OPTION)
    # eval_experiment_jsonl(EXPERIMENT_FILE, CKPT_PATH, aux_option=AUX_OPTION)
    # for k in [(0.186469,0.050000,0.084856),
    #       (0.050000,0.148778,0.162703),
    #       (0.050000,0.126131,0.190308),
    #       (0.185845,0.050000,0.159883)]:
    #     eval_approx_val_by_dims(k, CKPT_PATH, aux_option=AUX_OPTION)




# box1 = 0.186469	0.05	0.084856	aspect ratio: 0.268140398304266
# box2 = 0.05	0.148778	0.162703	aspect ratio: 0.307307667296103
# box3 = 0.05	0.126131	0.190308	aspect ratio: 0.262731907599988
# box4 = 0.185845	0.05	0.159883	aspect ratio: 0.269041448958189
