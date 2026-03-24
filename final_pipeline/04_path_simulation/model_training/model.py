# file: pickplace_dataset_and_model.py

import os, json
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------
# CONSTANTS (edit here)
# ---------------------
RAW_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
PAIRS_PATH       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs/pairs_train.jsonl"

# Table extents for normalization (meters)
TABLE_L, TABLE_W, TABLE_Z = 1.5, 0.6, 0.10

# Fixed box dims (meters) — used ONLY to scale ^Oi p_C (not fed as a feature)
BOX_DIMS = np.array([0.143, 0.0915, 0.051], dtype=np.float32)

# Feature toggles
USE_CORNERS = False  # default off; model supports corners if enabled later
USE_META    = False   # include grasp metadata: [u_frac, v_frac, one-hot(+X,-X,+Y,-Y)]
USE_DELTA   = False  # if True: replace place_abs with world delta features

# Ablation toggles (transport only; rotations fixed to 6D)

# If True, express the grasp in the transport frame using the pick->place
# relative rotation (R_delta): translate by R_delta @ t_loc and rotate by
# R_delta @ R_OiC. Only affects the grasp encoding.
USE_TRANSPORT = False

# Corners normalization strategy when USE_CORNERS=True
#   - "layernorm": feed raw corners; rely on model LayerNorm
#   - "zscore"   : dataset-level z-score per split (streamed; stats cached to disk)
CORNERS_NORM = "layernorm"

# Bound the number of raw JSON files cached simultaneously
MAX_RAW_CACHE = 256


# ---------------------
# Math helpers (numpy)
# ---------------------
def wxyz_to_R(qwxyz):
    w, x, y, z = qwxyz
    # normalized quaternion → rotation matrix (world ← local)
    n = w*w + x*x + y*y + z*z
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1.0 - (yy + zz),     xy - wz,           xz + wy],
        [    xy + wz,     1.0 - (xx + zz),       yz - wx],
        [    xz - wy,         yz + wx,       1.0 - (xx + yy)]
    ], dtype=np.float32)
    return R

def rot_to_6d(Rm):
    # Zhou et al. 6D: first two columns
    return np.concatenate([Rm[:, 0], Rm[:, 1]], axis=0).astype(np.float32)

# (Quaternion helpers removed; we use fixed 6D representation)

def corners_world(pos_w, R_w, dims):
    # 8 corners at ±half extents, world = pos + R * corner
    hx, hy, hz = 0.5 * dims
    cs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                c = np.array([sx*hx, sy*hy, sz*hz], dtype=np.float32)
                p = pos_w + R_w @ c
                cs.append(p)
    return np.concatenate(cs, axis=0).astype(np.float32)  # 8*3=24

def norm_pos_world(p):
    return np.array([p[0]/TABLE_L, p[1]/TABLE_W, p[2]/TABLE_Z], dtype=np.float32)

def face_one_hot(face):
    # only +X, -X, +Y, -Y are used
    mapping = {"+X":0, "-X":1, "+Y":2, "-Y":3}
    v = np.zeros(4, dtype=np.float32)
    idx = mapping.get(str(face), None)
    if idx is not None: v[idx] = 1.0
    return v


# ---------------------
# Dataset
# ---------------------
class PickPlaceDataset(Dataset):
    """
    Loads pairs_*.jsonl and resolves pick/place endpoints into features:

      grasp_Oi   : 9  = [ ^Oi p_C (3 scaled by BOX_DIMS), 6D(R_OiC) ]
      objW_pick  : 9  = [ ^W p_Oi (3 norm), 6D(R_WOi) ]
      objW_place : 9  = [ ^W p_Of (3 norm), 6D(R_WOf) ]   (or world delta if USE_DELTA)
      meta       : 6  = [ u_frac, v_frac, one-hot(+X,-X,+Y,-Y) ]     (optional)
      corners_f  : 24 = z-scored corners at final pose               (optional)

      y_ik, y_col: binary labels from JSONL
    """
    def __init__(self, pairs_path, raw_root, grasps_meta_path,
                 use_corners=USE_CORNERS, use_meta=USE_META, use_delta=USE_DELTA,
                 precomp_dir=None):
        self.use_corners = use_corners
        self.use_meta = use_meta
        self.use_delta = use_delta
        self.precomp_dir = precomp_dir

        if self.precomp_dir is not None:
            # Fast path: load memory-mapped arrays produced by precompute.py
            with open(os.path.join(self.precomp_dir, "meta.json"), "r") as f:
                info = json.load(f)
            self.n = int(info["n"]) if "n" in info else None

            self.mm = {}
            self.mm["grasp_Oi"]   = np.load(os.path.join(self.precomp_dir, "grasp_Oi.npy"), mmap_mode='r')
            self.mm["objW_pick"]  = np.load(os.path.join(self.precomp_dir, "objW_pick.npy"), mmap_mode='r')
            self.mm["objW_place"] = np.load(os.path.join(self.precomp_dir, "objW_place.npy"), mmap_mode='r')
            if self.use_meta and os.path.exists(os.path.join(self.precomp_dir, "meta.npy")):
                self.mm["meta"] = np.load(os.path.join(self.precomp_dir, "meta.npy"), mmap_mode='r')
            if self.use_corners and os.path.exists(os.path.join(self.precomp_dir, "corners_f.npy")):
                self.mm["corners_f"] = np.load(os.path.join(self.precomp_dir, "corners_f.npy"), mmap_mode='r')
            self.mm["y_ik"]  = np.load(os.path.join(self.precomp_dir, "y_ik.npy"), mmap_mode='r')
            self.mm["y_col"] = np.load(os.path.join(self.precomp_dir, "y_col.npy"), mmap_mode='r')

            if self.n is None:
                self.n = self.mm["y_ik"].shape[0]

            self.view = {
                "grasp_Oi":   self.mm["grasp_Oi"].reshape(self.n, self.mm["grasp_Oi"].shape[-1]),
                "objW_pick":  self.mm["objW_pick"].reshape(self.n, self.mm["objW_pick"].shape[-1]),
                "objW_place": self.mm["objW_place"].reshape(self.n, self.mm["objW_place"].shape[-1]),
                "y_ik":       self.mm["y_ik"].reshape(self.n, 1),
                "y_col":      self.mm["y_col"].reshape(self.n, 1),
            }
            if "meta" in self.mm:
                self.view["meta"] = self.mm["meta"].reshape(self.n, 6)
            if "corners_f" in self.mm:
                self.view["corners_f"] = self.mm["corners_f"].reshape(self.n, 24)

            # Raw path attributes kept minimal to avoid branching elsewhere
            self.pairs = None
            self.gmeta = None
            self.raw_root = raw_root
            self.cache = None
            self.pairs_path = pairs_path
            self.corners_mean = None
            self.corners_std = None
        else:
            # Original path: stream JSON lines and raw JSON files
            self.pairs = []
            with open(pairs_path, "r") as f:
                for line in f:
                    self.pairs.append(json.loads(line))

            with open(grasps_meta_path, "r") as f:
                raw = json.load(f)
            self.gmeta = {int(k): v for k, v in raw.items()}

            self.raw_root = raw_root
            self.cache = OrderedDict()  # (p,g) -> loaded dict from raw JSON (LRU)
            self.pairs_path = pairs_path

            # Corners normalization configuration
            self.corners_mean = None
            self.corners_std = None
            if self.use_corners and CORNERS_NORM == "zscore":
                base = os.path.splitext(os.path.basename(self.pairs_path))[0]
                stats_path = os.path.join(os.path.dirname(self.pairs_path), f"{base}_corners_stats.json")

                if os.path.exists(stats_path):
                    with open(stats_path, "r") as f:
                        stats = json.load(f)
                    self.corners_mean = np.array(stats["mean"], dtype=np.float32)
                    self.corners_std  = np.array(stats["std"], dtype=np.float32)
                else:
                    n = 0
                    mean = np.zeros(24, dtype=np.float32)
                    M2 = np.zeros(24, dtype=np.float32)
                    for rec in self.pairs:
                        p2, g2, o2 = rec["place"]["p"], rec["place"]["g"], rec["place"]["o"]
                        place_row = self._row(p2, g2, o2)
                        pos_f = np.array(place_row["object_pose_world"]["position"], dtype=np.float32)
                        quat_f = np.array(place_row["object_pose_world"]["orientation_quat"], dtype=np.float32)
                        R_f = wxyz_to_R(quat_f)
                        cf = corners_world(pos_f, R_f, BOX_DIMS)
                        n += 1
                        delta = cf - mean
                        mean = mean + delta / float(n)
                        delta2 = cf - mean
                        M2 = M2 + delta * delta2
                        if len(self.cache) > MAX_RAW_CACHE:
                            self.cache.popitem(last=False)

                    var = (M2 / max(1, n - 1)).astype(np.float32)
                    std = np.sqrt(var) + 1e-8
                    self.corners_mean = mean.astype(np.float32)
                    self.corners_std = std.astype(np.float32)

                    with open(stats_path, "w") as f:
                        json.dump({"mean": self.corners_mean.tolist(), "std": self.corners_std.tolist()}, f)

    def _file(self, p, g):
        return os.path.join(self.raw_root, f"p{int(p)}", f"data_{int(g)}.json")

    def _row(self, p, g, o):
        key = (int(p), int(g))
        if key in self.cache:
            # Mark as recently used
            self.cache.move_to_end(key, last=True)
        else:
            with open(self._file(p, g), "r") as f:
                data = json.load(f)
            self.cache[key] = data
            # Evict least-recently-used if over capacity
            if len(self.cache) > MAX_RAW_CACHE:
                self.cache.popitem(last=False)
        return self.cache[key][str(int(o))]

    def __len__(self):
        if self.precomp_dir is not None:
            return self.n
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.precomp_dir is not None:
            out = {
                "grasp_Oi":   torch.from_numpy(self.view["grasp_Oi"][idx].copy()),
                "objW_pick":  torch.from_numpy(self.view["objW_pick"][idx].copy()),
                "objW_place": torch.from_numpy(self.view["objW_place"][idx].copy()),
                "y_ik":       torch.from_numpy(self.view["y_ik"][idx].copy()).float(),
                "y_col":      torch.from_numpy(self.view["y_col"][idx].copy()).float(),
            }
            if "meta" in self.view:
                out["meta"] = torch.from_numpy(self.view["meta"][idx].copy())
            if "corners_f" in self.view:
                out["corners_f"] = torch.from_numpy(self.view["corners_f"][idx].copy())
            return out
        else:
            rec = self.pairs[idx]
            p1, g,  o1 = rec["pick"]["p"],  rec["pick"]["g"],  rec["pick"]["o"]
            p2, g2, o2 = rec["place"]["p"], rec["place"]["g"], rec["place"]["o"]

            pick  = self._row(p1, g,  o1)
            place = self._row(p2, g2, o2)

            pos_i  = np.array(pick["object_pose_world"]["position"], dtype=np.float32)
            quat_i = np.array(pick["object_pose_world"]["orientation_quat"], dtype=np.float32)
            R_WOi  = wxyz_to_R(quat_i)

            pos_f  = np.array(place["object_pose_world"]["position"], dtype=np.float32)
            quat_f = np.array(place["object_pose_world"]["orientation_quat"], dtype=np.float32)
            R_WOf  = wxyz_to_R(quat_f)

            posCi  = np.array(pick["grasp_pose_contact_world"]["position"], dtype=np.float32)
            quatCi = np.array(pick["grasp_pose_contact_world"]["orientation_quat"], dtype=np.float32)
            R_WCi  = wxyz_to_R(quatCi)

            t_loc = R_WOi.T @ (posCi - pos_i)
            t_loc = (t_loc / BOX_DIMS).astype(np.float32)
            R_OiC = R_WOi.T @ R_WCi

            R_delta = R_WOf @ R_WOi.T

            if USE_TRANSPORT:
                t_for_grasp = (R_delta @ t_loc).astype(np.float32)
                R_grasp = R_delta @ R_OiC
            else:
                t_for_grasp = t_loc
                R_grasp = R_OiC

            grasp_Oi  = np.concatenate([t_for_grasp, rot_to_6d(R_grasp)], axis=0)
            objW_pick = np.concatenate([norm_pos_world(pos_i), rot_to_6d(R_WOi)], axis=0)
            if self.use_delta:
                t_delta = norm_pos_world(pos_f - pos_i)
                place_feat = np.concatenate([t_delta, rot_to_6d(R_delta)], axis=0)
            else:
                place_feat = np.concatenate([norm_pos_world(pos_f), rot_to_6d(R_WOf)], axis=0)

            meta = None
            if self.use_meta:
                gm = self.gmeta[int(g)]
                meta = np.concatenate([
                    np.array([float(gm.get("u_frac", 0.5)), float(gm.get("v_frac", 0.5))], dtype=np.float32),
                    face_one_hot(gm.get("face", None))
                ], axis=0)

            corners_f = None
            if self.use_corners:
                cf = corners_world(pos_f, R_WOf, BOX_DIMS)
                if CORNERS_NORM == "zscore" and self.corners_mean is not None and self.corners_std is not None:
                    cf = (cf - self.corners_mean) / self.corners_std
                corners_f = cf.astype(np.float32)

            y_ik  = np.float32(rec["labels"]["ik"])
            y_col = np.float32(rec["labels"]["collision"])

            out = {
                "grasp_Oi":   torch.from_numpy(grasp_Oi),
                "objW_pick":  torch.from_numpy(objW_pick),
                "objW_place": torch.from_numpy(place_feat),
                "y_ik":       torch.tensor([y_ik], dtype=torch.float32),
                "y_col":      torch.tensor([y_col], dtype=torch.float32),
            }
            if meta is not None:
                out["meta"] = torch.from_numpy(meta)
            if corners_f is not None:
                out["corners_f"] = torch.from_numpy(corners_f)
            return out


# ---------------------
# Model
# ---------------------
def mlp(d_in, dims, dropout=0.05):
    layers = []
    prev = d_in
    for d in dims:
        layers += [nn.Linear(prev, d), nn.GELU(), nn.Dropout(dropout)]
        prev = d
    return nn.Sequential(*layers)

class PickPlaceFeasibilityNet(nn.Module):
    """
    Two-head MLP aligned to frame-based features.

    forward(
      grasp_Oi   : [B,  9],
      objW_pick  : [B,  9],
      objW_place : [B,  9],     # or world delta
      meta       : [B,  6]?     # optional
      corners_f  : [B, 24]?     # optional (z-scored)
    ) -> logit_ik [B,1], logit_collision [B,1]
    """
    def __init__(self, use_meta=USE_META, use_corners=USE_CORNERS, hidden=64, dropout=0.05):
        super().__init__()
        self.use_meta = use_meta
        self.use_corners = use_corners
        d_pose = 9

        self.enc_grasp  = mlp(d_pose,  [hidden, hidden], dropout)
        self.enc_pick   = mlp(d_pose,  [hidden, hidden], dropout)
        self.enc_place  = mlp(d_pose,  [hidden, hidden], dropout)
        if self.use_meta:
            self.enc_meta   = mlp(6,  [16, 16], dropout)
        if self.use_corners:
            self.enc_corn   = mlp(24, [128, hidden], dropout)
            self.c_ln       = nn.LayerNorm(hidden)

        fuse_in = hidden + hidden + hidden
        if self.use_meta:    fuse_in += 16
        if self.use_corners: fuse_in += hidden

        self.fuse = mlp(fuse_in, [hidden*2, hidden], dropout)
        self.fuse_ln = nn.LayerNorm(hidden)
        self.head_ik  = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.head_col = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, grasp_Oi, objW_pick, objW_place, meta=None, corners_f=None):
        eg = self.enc_grasp(grasp_Oi)
        ep = self.enc_pick(objW_pick)
        el = self.enc_place(objW_place)
        feats = [eg, ep, el]

        if self.use_meta and meta is not None:
            em = self.enc_meta(meta)
            feats.append(em)
        if self.use_corners and corners_f is not None:
            ec = self.enc_corn(corners_f)
            ec = self.c_ln(ec)
            feats.append(ec)

        h = torch.cat(feats, dim=-1)
        z = self.fuse(h)
        z = self.fuse_ln(z)
        logit_ik  = self.head_ik(z)
        logit_col = self.head_col(z)
        return logit_ik, logit_col


# ---------------------
# Tiny sanity (no training)
# ---------------------
if __name__ == "__main__":
    USE_META = False
    USE_CORNERS = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/training_out/no_meta/best.pt"
    model = PickPlaceFeasibilityNet(use_meta=USE_META, use_corners=USE_CORNERS, hidden=64, dropout=0.05)
    data = {
        "grasp_Oi": torch.randn(1, 9),
        "objW_pick": torch.randn(1, 9),
        "objW_place": torch.randn(1, 9),
        "corners_f": torch.randn(1, 24),
    }
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    model.eval()
