# collision_dataset_v6.py
# - Builds memmaps for train/val from combined_data JSON (expects "initial_object_pose").
# - Base features (unchanged): corners_24[z], aux_12 = [t_loc_z(3), R_loc6(6 raw), dims_z(3)], label (collision).
# - Optional extras (toggle with FEATURE_OPTION):
#     1 = BASE
#     2 = BASE + R_of6            (init→final rotation, 6D)
#     3 = BASE + t_loc_final2     (t_loc in final frame, in-plane normalized, 2D)
#     4 = BASE + R_hf6            (hand→final rotation, 6D)  with R_hf = R_of @ R_loc
#
# IMPORTANT: Options 2–4 REQUIRE "initial_object_pose" in the JSON.
# If missing, this code will error – regenerate combined_data to include it.

import os, json, numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# ----------------------------- CONFIG -----------------------------
DATA_ROOT    = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
TRAIN_JSON   = os.path.join(DATA_ROOT, "combined_data", "train.json")
VAL_JSON     = os.path.join(DATA_ROOT, "combined_data", "val.json")
TRAIN_MM     = os.path.join(DATA_ROOT, "memmaps_train")
VAL_MM       = os.path.join(DATA_ROOT, "memmaps_val")
TEST_JSON    = os.path.join(DATA_ROOT, "combined_data", "test.json")
TEST_MM      = os.path.join(DATA_ROOT, "memmaps_test")

# Choose which feature set this dataset should expose:
#   1=BASE, 2=BASE+R_of6, 3=BASE+t_loc_final2, 4=BASE+R_hf6
FEATURE_OPTION = 2

# --------------------------- IO HELPERS ---------------------------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def _stream_array(path: str):
    dec = json.JSONDecoder()
    buf, i, in_arr = "", 0, False
    CHUNK = 4*1024*1024
    with open(path, "r") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                while True:
                    while i < len(buf) and buf[i] in " \r\n\t,":
                        i += 1
                    if i >= len(buf): return
                    if not in_arr:
                        j = buf.find("[", i)
                        if j < 0: return
                        in_arr, i = True, j+1
                        continue
                    if i < len(buf) and buf[i] == "]": return
                    try:
                        obj, end = dec.raw_decode(buf, i)
                    except json.JSONDecodeError:
                        return
                    yield obj
                    i = end
            else:
                buf += chunk
                if len(buf) > 8*CHUNK:
                    buf = buf[i:]; i = 0
                while True:
                    while i < len(buf) and buf[i] in " \r\n\t,":
                        i += 1
                    if i >= len(buf): break
                    if not in_arr:
                        j = buf.find("[", i)
                        if j < 0: break
                        in_arr, i = True, j+1
                        continue
                    if buf[i] == "]": return
                    try:
                        obj, end = dec.raw_decode(buf, i)
                    except json.JSONDecodeError:
                        break
                    yield obj
                    i = end

# ---------------------- GEOMETRY (VECTORIZED) ---------------------
def quat_wxyz_to_R_batched(q: np.ndarray) -> np.ndarray:
    # q: (B,4) [w,x,y,z] -> (B,3,3) world-of-object axes as columns
    w,x,y,z = q[:,0], q[:,1], q[:,2], q[:,3]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w/=n; x/=n; y/=n; z/=n
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    R = np.empty((q.shape[0],3,3), dtype=np.float32)
    R[:,0,0] = 1 - 2*(yy+zz); R[:,0,1] = 2*(xy - wz);   R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz);   R[:,1,1] = 1 - 2*(xx+zz); R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy);   R[:,2,1] = 2*(yz + wx);   R[:,2,2] = 1 - 2*(xx+yy)
    return R

def r6_to_R_batched(r6: np.ndarray) -> np.ndarray:
    # r6: (B,6) -> (B,3,3), Zhou 6D to full R via Gram-Schmidt
    a1 = r6[:,0:3]; a2 = r6[:,3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2p= a2 - (np.sum(b1*a2, axis=1, keepdims=True))*b1
    b2 = a2p / np.linalg.norm(a2p, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1,b2,b3], axis=2)  # columns

SIGNS_8x3 = np.array(
    [[-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
     [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]], dtype=np.float32)

def corners_world_from_dims_final_batch(dims: np.ndarray, final7: np.ndarray) -> np.ndarray:
    B = dims.shape[0]
    T = final7[:, :3].astype(np.float32)
    q = final7[:, 3:].astype(np.float32)
    R = quat_wxyz_to_R_batched(q)
    C_obj = SIGNS_8x3[None,:,:] * (0.5*dims)[:,None,:]
    C_w = np.einsum("bij,bkj->bki", R, C_obj) + T[:,None,:]
    return C_w.reshape(B,24)

def up_axis_idx_from_Rf(Rf: np.ndarray) -> np.ndarray:
    # Rf: (B,3,3). Choose which object axis (X=0,Y=1,Z=2) is most aligned with world +Z, ignoring sign.
    zcols = np.abs(Rf[:,2,:])      # (B,3): z-component of each column
    return np.argmax(zcols, axis=1)  # (B,)

# ----------------------- MEMMAP BUILDING --------------------------
def _count_total(path: str) -> int:
    c=0
    for _ in tqdm(_stream_array(path), desc=f"Counting {os.path.basename(path)}", unit="obj", dynamic_ncols=True): c+=1
    return c

def build_memmaps(in_json: str, out_dir: str, batch: int = 200_000):
    ensure_dir(out_dir)
    N = _count_total(in_json)

    # Preallocate memmaps
    mm_dims   = np.memmap(os.path.join(out_dir, "dims3.mm"),   dtype=np.float32, mode="w+", shape=(N,3))
    mm_tloc   = np.memmap(os.path.join(out_dir, "tloc3.mm"),   dtype=np.float32, mode="w+", shape=(N,3))
    mm_rloc6  = np.memmap(os.path.join(out_dir, "rloc6.mm"),   dtype=np.float32, mode="w+", shape=(N,6))
    mm_final7 = np.memmap(os.path.join(out_dir, "final7.mm"),  dtype=np.float32, mode="w+", shape=(N,7))
    mm_init7  = np.memmap(os.path.join(out_dir, "init7.mm"),   dtype=np.float32, mode="w+", shape=(N,7))  # NEW
    mm_c24    = np.memmap(os.path.join(out_dir, "corners24.mm"), dtype=np.float32, mode="w+", shape=(N,24))
    mm_y      = np.memmap(os.path.join(out_dir, "label1.mm"), dtype=np.float32, mode="w+", shape=(N,1))

    # extras
    mm_rof6   = np.memmap(os.path.join(out_dir,"rof6.mm"),    dtype=np.float32, mode="w+", shape=(N,6))
    mm_tlfin2 = np.memmap(os.path.join(out_dir,"tlocf2.mm"),  dtype=np.float32, mode="w+", shape=(N,2))
    mm_rhf6   = np.memmap(os.path.join(out_dir,"rhf6.mm"),    dtype=np.float32, mode="w+", shape=(N,6))

    w=0
    buf_dims, buf_final, buf_init, buf_tloc, buf_rloc6, buf_y = [], [], [], [], [], []
    pbar = tqdm(total=N, desc="Building memmaps", unit="obj", dynamic_ncols=True, smoothing=0.1)


    for obj in _stream_array(in_json):
        # REQUIRED KEYS (no guards)
        buf_dims.append(obj["object_dimensions"])
        buf_final.append(obj["final_object_pose"])      # [tx,ty,tz,qw,qx,qy,qz]
        buf_init.append(obj["init_object_pose"])     # [tx,ty,tz,qw,qx,qy,qz]   <-- must exist
        buf_tloc.append(obj["t_loc"])                   # [3]
        buf_rloc6.append(obj["R_loc6"])                 # [6]
        y = 1.0 if float(obj["collision_label"]) > 0.5 else 0.0
        buf_y.append([y])
        if len(buf_dims) == batch:
            _flush_batch(w, mm_dims, mm_final7, mm_init7, mm_tloc, mm_rloc6, mm_y, mm_c24, mm_rof6, mm_tlfin2, mm_rhf6,
                         np.asarray(buf_dims,np.float32),
                         np.asarray(buf_final,np.float32),
                         np.asarray(buf_init,np.float32),
                         np.asarray(buf_tloc,np.float32),
                         np.asarray(buf_rloc6,np.float32),
                         np.asarray(buf_y,np.float32))
            w += batch
            buf_dims.clear(); buf_final.clear(); buf_init.clear(); buf_tloc.clear(); buf_rloc6.clear(); buf_y.clear()
        pbar.update(1)

    if buf_dims:
        _flush_batch(w, mm_dims, mm_final7, mm_init7, mm_tloc, mm_rloc6, mm_y, mm_c24, mm_rof6, mm_tlfin2, mm_rhf6,
                     np.asarray(buf_dims,np.float32),
                     np.asarray(buf_final,np.float32),
                     np.asarray(buf_init,np.float32),
                     np.asarray(buf_tloc,np.float32),
                     np.asarray(buf_rloc6,np.float32),
                     np.asarray(buf_y,np.float32))
    pbar.close()

    # flush files
    for mm in (mm_dims,mm_final7,mm_init7,mm_tloc,mm_rloc6,mm_y,mm_c24,mm_rof6,mm_tlfin2,mm_rhf6): mm.flush()

    meta = {
        "N": N,
        "dims_file":   os.path.join(out_dir,"dims3.mm"),
        "tloc_file":   os.path.join(out_dir,"tloc3.mm"),
        "rloc6_file":  os.path.join(out_dir,"rloc6.mm"),
        "final7_file": os.path.join(out_dir,"final7.mm"),
        "init7_file":  os.path.join(out_dir,"init7.mm"),
        "corners24_file": os.path.join(out_dir,"corners24.mm"),
        "label_file":  os.path.join(out_dir,"label1.mm"),
        "rof6_file":   os.path.join(out_dir,"rof6.mm"),     # for option 2
        "tlocf2_file": os.path.join(out_dir,"tlocf2.mm"),   # for option 3
        "rhf6_file":   os.path.join(out_dir,"rhf6.mm"),     # for option 4
    }
    with open(os.path.join(out_dir,"meta.json"),"w") as f: json.dump(meta,f)
    print(f"[memmap] wrote {N} rows to {out_dir}")

def _flush_batch(w, mm_dims, mm_final7, mm_init7, mm_tloc, mm_rloc6, mm_y, mm_c24, mm_rof6, mm_tlfin2, mm_rhf6,
                 dims_b, final_b, init_b, tloc_b, rloc6_b, y_b):
    B = dims_b.shape[0]
    # base
    c24_b = corners_world_from_dims_final_batch(dims_b, final_b)
    mm_dims[w:w+B]   = dims_b
    mm_final7[w:w+B] = final_b
    mm_init7[w:w+B]  = init_b
    mm_tloc[w:w+B]   = tloc_b
    mm_rloc6[w:w+B]  = rloc6_b
    mm_c24[w:w+B]    = c24_b
    mm_y[w:w+B]      = y_b
    # extras (vectorized)
    Rf = quat_wxyz_to_R_batched(final_b[:,3:7])             # (B,3,3)
    Ro = quat_wxyz_to_R_batched(init_b[:,3:7])              # (B,3,3)
    R_of = np.einsum("bij,bjk->bik", np.transpose(Rf,(0,2,1)), Ro)  # R_f^T R_o
    # Pack first two columns as [col0; col1] (not row-interleaved)
    rof6_b = np.concatenate([R_of[:,:,0], R_of[:,:,1]], axis=1).astype(np.float32)
    mm_rof6[w:w+B] = rof6_b

    # in-plane t_loc in final frame
    tloc_f = np.einsum("bij,bj->bi", R_of, tloc_b)          # (B,3)
    up_idx = up_axis_idx_from_Rf(Rf)                        # (B,)
    # map (up=Z)->(X,Y), (up=X)->(Y,Z), (up=Y)->(X,Z)
    hx = np.empty(B, dtype=np.float32)
    hy = np.empty(B, dtype=np.float32)
    ux = np.empty(B, dtype=np.float32)
    uy = np.empty(B, dtype=np.float32)
    # up==2 (Z)
    m = (up_idx==2)
    hx[m] = 0.5*dims_b[m,0]; hy[m] = 0.5*dims_b[m,1]
    ux[m] = tloc_f[m,0]/hx[m]; uy[m] = tloc_f[m,1]/hy[m]
    # up==0 (X)
    m = (up_idx==0)
    hx[m] = 0.5*dims_b[m,1]; hy[m] = 0.5*dims_b[m,2]
    ux[m] = tloc_f[m,1]/hx[m]; uy[m] = tloc_f[m,2]/hy[m]
    # up==1 (Y)
    m = (up_idx==1)
    hx[m] = 0.5*dims_b[m,0]; hy[m] = 0.5*dims_b[m,2]
    ux[m] = tloc_f[m,0]/hx[m]; uy[m] = tloc_f[m,2]/hy[m]
    u2 = np.stack([np.clip(ux,-1,1), np.clip(uy,-1,1)], axis=1).astype(np.float32)
    mm_tlfin2[w:w+B] = u2

    # R_hf = R_f^T R_h = (R_f^T R_o) (R_o^T R_h) = R_of @ R_loc
    R_loc = r6_to_R_batched(rloc6_b)                        # (B,3,3)
    R_hf  = np.einsum("bij,bjk->bik", R_of, R_loc)
    rhf6_b = np.concatenate([R_hf[:,:,0], R_hf[:,:,1]], axis=1).astype(np.float32)
    mm_rhf6[w:w+B] = rhf6_b

# ----------------------------- STATS ------------------------------
def compute_train_stats(mem_dir: str, sample_cap: int = 200_000) -> Dict[str, List[float]]:
    meta = json.load(open(os.path.join(mem_dir,"meta.json"),"r"))
    N = int(meta["N"])
    take = min(sample_cap, N)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, size=take, replace=False)
    mm_dims = np.memmap(meta["dims_file"], dtype=np.float32, mode="r", shape=(N,3))
    mm_tloc = np.memmap(meta["tloc_file"], dtype=np.float32, mode="r", shape=(N,3))
    mm_c24  = np.memmap(meta["corners24_file"], dtype=np.float32, mode="r", shape=(N,24))
    d = mm_dims[idx]; t = mm_tloc[idx]; c = mm_c24[idx]
    stats = {
        "corners_mean": c.mean(axis=0).astype(np.float32).tolist(),
        "corners_std":  (c.std(axis=0)+1e-12).astype(np.float32).tolist(),
        "tloc_mean":    t.mean(axis=0).astype(np.float32).tolist(),
        "tloc_std":     (t.std(axis=0)+1e-12).astype(np.float32).tolist(),
        "dims_mean":    d.mean(axis=0).astype(np.float32).tolist(),
        "dims_std":     (d.std(axis=0)+1e-12).astype(np.float32).tolist(),
    }
    with open(os.path.join(mem_dir,"stats.json"),"w") as f: json.dump(stats,f)
    print(f"[stats] saved to {os.path.join(mem_dir,'stats.json')} (take={take})")
    return stats

# ----------------------------- DATASET ----------------------------
class FinalCornersHandDataset(Dataset):
    """
    Returns:
      corners_24  : (24,) float32, z-scored with train stats
      aux_k       : (K,)  float32
          K by FEATURE_OPTION:
            1 -> 12 : [ t_loc_z(3), R_loc6(6 raw), dims_z(3) ]
            2 -> 18 : base + R_of6(6)
            3 -> 14 : base + t_loc_final2(2)
            4 -> 18 : base + R_hf6(6)
      label       : ()    float32  (1 if collision else 0)
    """
    def __init__(self, mem_dir: str, normalization_stats: Optional[Dict]=None, is_training: bool=True, feature_option: int=FEATURE_OPTION):
        meta = json.load(open(os.path.join(mem_dir,"meta.json"),"r"))
        self.N = int(meta["N"])
        self.mm_dims   = np.memmap(meta["dims_file"],    dtype=np.float32, mode="r", shape=(self.N,3))
        self.mm_tloc   = np.memmap(meta["tloc_file"],    dtype=np.float32, mode="r", shape=(self.N,3))
        self.mm_rloc6  = np.memmap(meta["rloc6_file"],   dtype=np.float32, mode="r", shape=(self.N,6))
        self.mm_c24    = np.memmap(meta["corners24_file"], dtype=np.float32, mode="r", shape=(self.N,24))
        self.mm_y      = np.memmap(meta["label_file"],   dtype=np.float32, mode="r", shape=(self.N,1))
        self.feature_option = int(feature_option)

        if normalization_stats is None:
            if is_training:
                normalization_stats = json.load(open(os.path.join(mem_dir,"stats.json"),"r"))
            else:
                raise RuntimeError("Provide train normalization_stats for eval.")
        self.ns = {
            "c_mu": torch.tensor(normalization_stats["corners_mean"], dtype=torch.float32),
            "c_sd": torch.tensor(normalization_stats["corners_std"],  dtype=torch.float32),
            "t_mu": torch.tensor(normalization_stats["tloc_mean"],    dtype=torch.float32),
            "t_sd": torch.tensor(normalization_stats["tloc_std"],     dtype=torch.float32),
            "d_mu": torch.tensor(normalization_stats["dims_mean"],    dtype=torch.float32),
            "d_sd": torch.tensor(normalization_stats["dims_std"],     dtype=torch.float32),
        }

        # extras as needed (loaded only when used)
        if self.feature_option == 2:
            self.mm_rof6 = np.memmap(meta["rof6_file"], dtype=np.float32, mode="r", shape=(self.N,6))
        elif self.feature_option == 3:
            self.mm_tlfin2 = np.memmap(meta["tlocf2_file"], dtype=np.float32, mode="r", shape=(self.N,2))
        elif self.feature_option == 4:
            self.mm_rhf6 = np.memmap(meta["rhf6_file"], dtype=np.float32, mode="r", shape=(self.N,6))

    def __len__(self): return self.N

    def __getitem__(self, i: int):
        c24 = torch.from_numpy(self.mm_c24[i].copy())   # (24,)
        t   = torch.from_numpy(self.mm_tloc[i].copy())  # (3,)
        r6  = torch.from_numpy(self.mm_rloc6[i].copy()) # (6,)
        d3  = torch.from_numpy(self.mm_dims[i].copy())  # (3,)
        y   = torch.tensor(float(self.mm_y[i,0]), dtype=torch.float32)

        c24 = (c24 - self.ns["c_mu"]) / (self.ns["c_sd"] + 1e-8)
        t_z = (t   - self.ns["t_mu"]) / (self.ns["t_sd"] + 1e-8)
        d_z = (d3  - self.ns["d_mu"]) / (self.ns["d_sd"] + 1e-8)
        aux = torch.cat([t_z, r6, d_z], dim=0)  # base 12

        if self.feature_option == 2:
            rof6 = torch.from_numpy(self.mm_rof6[i].copy())   # (6,)
            aux = torch.cat([aux, rof6], dim=0)               # 18
        elif self.feature_option == 3:
            u2 = torch.from_numpy(self.mm_tlfin2[i].copy())   # (2,) already in [-1,1]
            aux = torch.cat([aux, u2], dim=0)                 # 14
        elif self.feature_option == 4:
            rhf6 = torch.from_numpy(self.mm_rhf6[i].copy())   # (6,)
            aux = torch.cat([aux, rhf6], dim=0)               # 18

        return c24.to(torch.float32), aux.to(torch.float32), y

# --------------------------- QUICK USAGE --------------------------
if __name__ == "__main__":
    # Build val then train (matching your usual order)
    ensure_dir(VAL_MM);   build_memmaps(VAL_JSON,   VAL_MM);   compute_train_stats(VAL_MM,   200_000)
    ensure_dir(TEST_MM); build_memmaps(TEST_JSON, TEST_MM); compute_train_stats(TEST_MM, 200_000)
    ensure_dir(TRAIN_MM); build_memmaps(TRAIN_JSON, TRAIN_MM); compute_train_stats(TRAIN_MM, 200_000)
    # Light sanity
    ds = FinalCornersHandDataset(TRAIN_MM, normalization_stats=None, is_training=True, feature_option=FEATURE_OPTION)
    x, a, y = ds[0]
    print(f"[sanity] feature_option={FEATURE_OPTION} -> corners {tuple(x.shape)} | aux {tuple(a.shape)} | y {float(y)}")
