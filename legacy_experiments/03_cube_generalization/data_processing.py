import os, glob, json, math, random, hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

# ============================ CONFIG ============================

DATA_DIR           = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
RAW_SUBDIR         = "raw_data"
PROCESSED_SUBDIR   = "processed_data"
COMBINED_SUBDIR    = "combined_data"

SOURCE_GLOB        = "object_*.json"

SEED               = 12345
rng_py             = random.Random(SEED)
rng_np             = np.random.default_rng(SEED)

# Per-pick quotas (exact per bucket)
QUOTAS = {
    "SMALL":    2,   # same face, Δθ ≤ THRESH_SMALL_DEG
    "MEDIUM":   3,   # same face, Δθ > THRESH_SMALL_DEG
    "ADJACENT": 3,   # different face (not opposite)
    "OPPOSITE": 2,   # opposite face
}

BUCKET_TARGET = {
    "ADJACENT": 0.667,
    "OPPOSITE": 0.167,
    "MEDIUM":   0.130,
    "SMALL":    0.036,
}

TARGET_SUCCESS_RATE_BY_BUCKET = {
    # Chosen so sum(BUCKET_TARGET[b] * p_b) ~= 0.478
    "ADJACENT": 0.408,
    "OPPOSITE": 0.558,
    "MEDIUM":   0.658,
    "SMALL":    0.758,
}

GROUP_CAP          = sum(QUOTAS.values())   # total per clean pick
THRESH_SMALL_DEG   = 45.0

# Target positive rate per bucket (pos = success==1 & collision==0)
TARGET_SUCCESS_RATE = 0.478

# Split fractions
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

# Opposite mapping (from your face-id rule)
_OPPOSITE = {1:3, 3:1, 2:4, 4:2, 5:6, 6:5}

# Signature rounding for regrouping (grasp identity)
T_ROUND = 1e-4
R_ROUND = 1e-4

# ============================ HELPERS ============================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def quat_normalize(qs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(qs, axis=1, keepdims=True)
    return qs / n

def quat_angle_deg_abs(qi: np.ndarray, qs: np.ndarray) -> np.ndarray:
    dots = np.abs(qs @ qi)              # |dot| for double-cover
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))

def grasp_signature(t_loc: List[float], R_loc6: List[float]) -> Tuple:
    # Round to form a stable hashable key
    return tuple([round(float(x), 4) for x in (t_loc + R_loc6)])

def sample_bucket_indices(idx_list: np.ndarray,
                          succ_arr: np.ndarray,
                          coll_arr: np.ndarray,
                          k: int,
                          target_pos_rate: float) -> np.ndarray:
    if k <= 0 or idx_list.size == 0:
        return np.empty((0,), dtype=np.int64)

    # Partition within idx_list
    pos_mask = (succ_arr[idx_list] == 1) & (coll_arr[idx_list] == 0)
    col_mask = (coll_arr[idx_list] == 1)
    # neutrals: fail & no-collision
    neu_mask = (~pos_mask) & (~col_mask)

    pos_idx = idx_list[pos_mask]
    col_idx = idx_list[col_mask]
    neu_idx = idx_list[neu_mask]

    n_pos = int(round(k * target_pos_rate))
    if n_pos > pos_idx.size:
        n_pos = pos_idx.size

    take = []
    if n_pos > 0:
        take.append(rng_np.choice(pos_idx, size=n_pos, replace=False))

    n_neg = k - n_pos
    if n_neg > 0:
        use_col = min(col_idx.size, n_neg)
        if use_col > 0:
            take.append(rng_np.choice(col_idx, size=use_col, replace=False))
        rem = n_neg - use_col
        if rem > 0 and neu_idx.size > 0:
            take.append(rng_np.choice(neu_idx, size=min(rem, neu_idx.size), replace=False))

    if len(take) == 0:
        return np.empty((0,), dtype=np.int64)

    out = np.concatenate(take, axis=0)
    if out.size < k:
        pool = np.setdiff1d(idx_list, out, assume_unique=False)
        if pool.size > 0:
            fill = rng_np.choice(pool, size=min(k - out.size, pool.size), replace=False)
            out = np.concatenate([out, fill], axis=0)
    return out

# Target-driven allocator (largest-remainder) with availability constraints
def _compute_target_alloc(bucket_lists: Dict[str, np.ndarray],
                          group_cap: int) -> Dict[str, int]:
    order = ("ADJACENT","OPPOSITE","MEDIUM","SMALL")
    desired = {b: group_cap * BUCKET_TARGET[b] for b in order}
    floors  = {b: int(math.floor(desired[b])) for b in order}
    avail   = {b: bucket_lists[b].size for b in order}

    # clip base by availability
    alloc = {b: min(floors[b], avail[b]) for b in order}
    used = sum(alloc.values())
    max_total = min(group_cap, sum(avail.values()))

    # fractional parts for largest remainder distribution
    frac = {b: desired[b] - floors[b] for b in order}
    remainder_order = sorted(order, key=lambda x: (-frac[x], order.index(x)))

    # Distribute leftover deterministically by remainder order, respecting availability
    while used < max_total:
        changed = False
        for b in remainder_order:
            if used >= max_total:
                break
            if alloc[b] < avail[b]:
                alloc[b] += 1
                used += 1
                changed = True
        if not changed:
            break

    return alloc

# ============================ 1) BUILD RICH PAIRS ============================

def build_rich_pairs(data_dir: str = DATA_DIR,
                     source_subdir: str = RAW_SUBDIR,
                     source_glob: str = SOURCE_GLOB,
                     out_subdir: str = PROCESSED_SUBDIR,
                     quotas: Dict[str,int] = QUOTAS,
                     group_cap: int = GROUP_CAP,
                     small_thresh_deg: float = THRESH_SMALL_DEG,
                     target_succ: float = TARGET_SUCCESS_RATE):

    src_dir = os.path.join(data_dir, source_subdir)
    out_dir = os.path.join(data_dir, out_subdir)
    ensure_dir(out_dir)

    files = sorted(glob.glob(os.path.join(src_dir, source_glob)))
    print(f"[pairs] {len(files)} raw files in {src_dir}")

    global_counts = Counter()
    total_pairs, succ_pairs = 0, 0

    for path in tqdm(files, desc="building pairs (per-grasp)"):
        # ---- Load & flatten raw rows ----
        raw = json.load(open(path, "r"))
        rows = []
        for _, lst in raw.items():
            rows.extend(lst)  # concat orientation batches

        N = len(rows)
        # vector packs
        faces = np.empty(N, dtype=np.int16)
        quats = np.empty((N,4), dtype=np.float64)  # object orientation (wxyz)
        succ  = np.empty(N, dtype=np.int8)
        coll  = np.empty(N, dtype=np.int8)
        ghit  = np.empty(N, dtype=np.int8)
        phit  = np.empty(N, dtype=np.int8)
        bhit  = np.empty(N, dtype=np.int8)
        poses = [None]*N
        dims3 = np.empty((N,3), dtype=np.float32)
        tloc  = np.empty((N,3), dtype=np.float32)
        rloc6 = np.empty((N,6), dtype=np.float32)
        sigs  = [None]*N

        for j, c in enumerate(rows):
            qw,qx,qy,qz  = c[1][3:7]
            quats[j,:]   = (qw,qx,qy,qz)
            succ[j]      = 1 if c[2] else 0
            coll[j]      = 1 if c[3] else 0
            ghit[j]      = 1 if c[4] else 0
            phit[j]      = 1 if c[5] else 0
            bhit[j]      = 1 if c[6] else 0
            poses[j]     = c[1]
            ex           = c[7]
            faces[j]     = int(ex["final_face_id"])
            dims3[j,:]   = np.asarray(ex["dims"], dtype=np.float32)
            tloc[j,:]    = np.asarray(ex["t_loc"], dtype=np.float32)
            rloc6[j,:]   = np.asarray(ex["R_loc6"], dtype=np.float32)
            sigs[j]      = grasp_signature(ex["t_loc"], ex["R_loc6"])

        quats = quat_normalize(quats)

        # ---- Regroup by grasp signature ----
        groups = defaultdict(list)
        for idx, key in enumerate(sigs):
            groups[key].append(idx)

        out = {k: [] for k in (
            "t_loc_list", "R_loc6_list",
            "init_object_poses",            # NEW
            "final_object_poses", "final_face_ids",
            "success_labels", "collision_labels",
            "object_dimensions"
        )}

        # ---- Process each grasp-group ----
        for _, g_idx in groups.items():
            g_idx = np.asarray(g_idx, dtype=np.int64)

            # Clean picks: success & no-collision & no hits
            clean_mask = (succ[g_idx]==1) & (coll[g_idx]==0) & (ghit[g_idx]==0) & (phit[g_idx]==0) & (bhit[g_idx]==0)
            clean_ids  = g_idx[clean_mask]
            if clean_ids.size == 0:
                continue

            # Pre-pull arrays for speed
            g_faces = faces[g_idx]
            g_quats = quats[g_idx]
            g_succ  = succ[g_idx]
            g_coll  = coll[g_idx]
            g_poses = [poses[int(k)] for k in g_idx]
            g_dims  = dims3[g_idx]
            g_tloc  = tloc[g_idx]
            g_rloc6 = rloc6[g_idx]

            # Map from local index to global index
            # but we only need local arrays for masks/sampling
            for li, gi in enumerate(clean_ids):
                # local index of pick inside group
                pick_local = int(np.where(g_idx == gi)[0][0])

                fi = int(g_faces[pick_local])

                # candidate mask (exclude self)
                all_local = np.arange(g_idx.size, dtype=np.int64)
                not_self  = all_local != pick_local

                # relation masks
                same_mask = (g_faces == fi)
                opp_mask  = (g_faces == _OPPOSITE[fi])
                adj_mask  = (~same_mask) & (~opp_mask)

                # SAME split by angle around pick quaternion
                ang_deg = quat_angle_deg_abs(g_quats[pick_local], g_quats)
                small_mask  = same_mask & (ang_deg <= small_thresh_deg)
                medium_mask = same_mask & (~small_mask)

                bucket_lists = {
                    "SMALL":    all_local[small_mask & not_self],
                    "MEDIUM":   all_local[medium_mask & not_self],
                    "ADJACENT": all_local[adj_mask   & not_self],
                    "OPPOSITE": all_local[opp_mask   & not_self],
                }

                # target-driven allocation by BUCKET_TARGET (no binding caps)
                alloc = _compute_target_alloc(bucket_lists, group_cap)

                # per-bucket sampling with per-bucket success mix
                pick_t = g_tloc[pick_local].tolist()
                pick_R = g_rloc6[pick_local].tolist()

                for b in ("SMALL","MEDIUM","ADJACENT","OPPOSITE"):
                    k = alloc[b]
                    if k <= 0:
                        continue
                    idx_bucket = bucket_lists[b]
                    if idx_bucket.size == 0:
                        continue

                    # steer class mix inside each bucket with your existing success-aware sampler
                    target_succ_b = TARGET_SUCCESS_RATE_BY_BUCKET[b]
                    chosen_loc = sample_bucket_indices(idx_bucket, g_succ, g_coll, k, target_succ_b)
                    if chosen_loc.size == 0:
                        continue
                    
                    init_pose_pick = g_poses[pick_local]   # << anchor object's pose at the pick
                    
                    for j_local in chosen_loc.tolist():
                        pose_j = g_poses[j_local]          # final pose for this pair
                        dims_j = g_dims[j_local].tolist()
                        face_j = int(g_faces[j_local])
                        s_j    = float(g_succ[j_local])
                        c_j    = float(g_coll[j_local])

                        out["t_loc_list"].append(pick_t)
                        out["R_loc6_list"].append(pick_R)
                        out["init_object_poses"].append(init_pose_pick)   # NEW
                        out["final_object_poses"].append(pose_j)
                        out["final_face_ids"].append(face_j)
                        out["object_dimensions"].append(dims_j)
                        out["success_labels"].append(s_j)
                        out["collision_labels"].append(c_j)

                        global_counts[b] += 1
                        total_pairs += 1
                        if (s_j > 0.5) and (c_j < 0.5):
                            succ_pairs += 1

        # ---- Save processed file ----
        out_path = os.path.join(out_dir, f"processed_{os.path.basename(path)}")
        json.dump(out, open(out_path, "w"))

        n    = len(out["success_labels"])
        col  = sum(1 for x in out["collision_labels"] if x > 0.5)
        succ_rate = sum(1 for s,c in zip(out["success_labels"], out["collision_labels"]) if (s > 0.5 and c < 0.5)) / max(1,n)
        col_rate = sum(1 for x in out["collision_labels"] if x > 0.5) / max(1, n)
        bucket_counts = Counter(out["final_face_ids"])  # crude—face ids, not relations
        print(f"  {os.path.basename(out_path)}: {n} pairs | collision {col_rate:.3f}")


    if total_pairs > 0:
        succ_rate = succ_pairs / total_pairs
        print(f"[pairs] GLOBAL: pairs={total_pairs}  success_rate={succ_rate:.3f}  (target={TARGET_SUCCESS_RATE:.3f})")
        for k in ("SMALL","MEDIUM","ADJACENT","OPPOSITE"):
            n = global_counts[k]
            print(f"        {k:<8} {n:>8}  {n/total_pairs:6.3f}")

# ============================ 2) COMBINE (streamed) ============================

def combine_processed_data(data_dir: str = DATA_DIR,
                           processed_subdir: str = PROCESSED_SUBDIR,
                           out_subdir: str = COMBINED_SUBDIR):
    proc_dir = os.path.join(data_dir, processed_subdir)
    out_dir  = os.path.join(data_dir, out_subdir)
    ensure_dir(out_dir)

    files = sorted(glob.glob(os.path.join(proc_dir, "processed_object_*.json")))
    out_path = os.path.join(out_dir, "all_data.json")

    print(f"[combine] writing {out_path} from {len(files)} files")
    total_written, succ_total = 0, 0

    with open(out_path, "w") as out_f:
        out_f.write("[\n")
        first = True
        for fp in tqdm(files, desc="combining"):
            d = json.load(open(fp, "r"))
            n = len(d["t_loc_list"])
            for i in range(n):
                obj = {
                    "t_loc": d["t_loc_list"][i],
                    "R_loc6": d["R_loc6_list"][i],
                    "init_object_pose":  d["init_object_poses"][i],   # NEW
                    "final_object_pose": d["final_object_poses"][i],
                    "final_face_id": int(d["final_face_ids"][i]),
                    "object_dimensions": d["object_dimensions"][i],
                    "success_label": d["success_labels"][i],
                    "collision_label": d["collision_labels"][i]
                }
                if not first:
                    out_f.write(",\n")
                else:
                    first = False
                out_f.write(json.dumps(obj))
                total_written += 1
                if (obj["success_label"] > 0.5) and (obj["collision_label"] < 0.5):
                    succ_total += 1
        out_f.write("\n]")

    if total_written > 0:
        print(f"[combine] total={total_written}  success_rate={succ_total/total_written:.3f}")

# ============================ 3) SPLIT (hash-by-dims; streamed) ============================

def _dims_key(d: List[float], nd: int = 6) -> Tuple[float,float,float]:
    return (round(float(d[0]), nd), round(float(d[1]), nd), round(float(d[2]), nd))

def _stream_array(path: str):
    with open(path, "r") as f:
        in_arr = False
        in_str = False
        esc = False
        depth = 0
        buf = []
        while True:
            ch = f.read(1)
            if not ch: break
            if in_str:
                buf.append(ch)
                if esc: esc = False
                elif ch == '\\': esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"':
                in_str = True; buf.append(ch); continue
            if ch == '[':
                in_arr = True; continue
            if not in_arr: continue
            if ch == '{':
                buf = ['{']; depth = 1
                while True:
                    c = f.read(1)
                    if not c: break
                    buf.append(c)
                    if c == '"':
                        in_str = not in_str
                    elif not in_str:
                        if c == '{': depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                yield json.loads(''.join(buf))
                                break
                continue
            if ch == ']':
                return

def _count_split_file(path: str):
    n = succ = coll = 0
    dims_set = set()
    for obj in _stream_array(path):
        n += 1
        if obj["collision_label"] > 0.5: coll += 1
        if (obj["success_label"] > 0.5) and (obj["collision_label"] < 0.5): succ += 1
        dims_set.add(_dims_key(obj["object_dimensions"]))
    succ_rate = succ / n if n else 0.0
    coll_rate = coll / n if n else 0.0
    return {"n": n, "succ_rate": succ_rate, "coll_rate": coll_rate, "dims": dims_set}

def split_all_data(data_dir: str = DATA_DIR,
                   out_subdir: str = COMBINED_SUBDIR,
                   train_frac: float = TRAIN_FRAC,
                   val_frac: float = VAL_FRAC,
                   test_frac: float = TEST_FRAC,
                   seed: int = SEED):
    out_dir = os.path.join(data_dir, out_subdir)
    in_path = os.path.join(out_dir, "all_data.json")
    train_p = os.path.join(out_dir, "train.json")
    val_p   = os.path.join(out_dir, "val.json")
    test_p  = os.path.join(out_dir, "test.json")

    outs = {"train.json": open(train_p, "w"),
            "val.json":   open(val_p, "w"),
            "test.json":  open(test_p, "w")}
    for f in outs.values(): f.write("[\n")
    first = {k: True for k in outs}

    def pick_split_by_key(key_tuple):
        h = hashlib.md5((str(key_tuple) + f"|{seed}").encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        return "train.json" if u < train_frac else ("val.json" if u < train_frac + val_frac else "test.json")

    print(f"[split] streaming {in_path}")
    for obj in tqdm(_stream_array(in_path), desc="Splitting"):
        key = _dims_key(obj["object_dimensions"])
        tgt = pick_split_by_key(key)
        if not first[tgt]:
            outs[tgt].write(",\n")
        else:
            first[tgt] = False
        outs[tgt].write(json.dumps(obj))

    for k, f in outs.items():
        f.write("\n]"); f.close()
    print("[split] wrote train/val/test")

    train_stats = _count_split_file(train_p)
    val_stats   = _count_split_file(val_p)
    test_stats  = _count_split_file(test_p)

    print(f"train.json: n={train_stats['n']}  success_rate={train_stats['succ_rate']:.3f}  collision_rate={train_stats['coll_rate']:.3f}")
    print(f"val.json:   n={val_stats['n']}    success_rate={val_stats['succ_rate']:.3f}    collision_rate={val_stats['coll_rate']:.3f}")
    print(f"test.json:  n={test_stats['n']}   success_rate={test_stats['succ_rate']:.3f}   collision_rate={test_stats['coll_rate']:.3f}")

    leak_tv = len((train_stats["dims"] & val_stats["dims"]))
    leak_tt = len((train_stats["dims"] & test_stats["dims"]))
    leak_vt = len((val_stats["dims"]   & test_stats["dims"]))
    print(f"leaky buckets (train∩val): {leak_tv}")
    print(f"leaky buckets (train∩test): {leak_tt}")
    print(f"leaky buckets (val∩test): {leak_vt}")

# ============================ DIRECT: RAW -> SPLIT (streamed) ============================

def direct_split_from_raw(data_dir: str = DATA_DIR,
                          source_subdir: str = RAW_SUBDIR,
                          source_glob: str = SOURCE_GLOB,
                          out_subdir: str = COMBINED_SUBDIR,
                          quotas: Dict[str,int] = QUOTAS,
                          group_cap: int = GROUP_CAP,
                          small_thresh_deg: float = THRESH_SMALL_DEG,
                          train_frac: float = TRAIN_FRAC,
                          val_frac: float = VAL_FRAC,
                          test_frac: float = TEST_FRAC,
                          seed: int = SEED):
    src_dir = os.path.join(data_dir, source_subdir)
    out_dir = os.path.join(data_dir, out_subdir)
    ensure_dir(out_dir)

    train_p = os.path.join(out_dir, "train.json")
    val_p   = os.path.join(out_dir, "val.json")
    test_p  = os.path.join(out_dir, "test.json")

    outs = {"train.json": open(train_p, "w"),
            "val.json":   open(val_p, "w"),
            "test.json":  open(test_p, "w")}
    for f in outs.values(): f.write("[\n")
    first = {k: True for k in outs}

    def pick_split_by_key(key_tuple):
        h = hashlib.md5((str(key_tuple) + f"|{seed}").encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        return "train.json" if u < train_frac else ("val.json" if u < train_frac + val_frac else "test.json")

    files = sorted(glob.glob(os.path.join(src_dir, source_glob)))
    print(f"[direct] {len(files)} raw files in {src_dir}")
    print(f"[direct] writing splits to {out_dir}")

    global_counts = Counter()
    total_pairs, succ_pairs = 0, 0

    for path in tqdm(files, desc="building & splitting (per-grasp)"):
        raw = json.load(open(path, "r"))
        rows = []
        for _, lst in raw.items():
            rows.extend(lst)

        N = len(rows)
        faces = np.empty(N, dtype=np.int16)
        quats = np.empty((N,4), dtype=np.float64)
        succ  = np.empty(N, dtype=np.int8)
        coll  = np.empty(N, dtype=np.int8)
        ghit  = np.empty(N, dtype=np.int8)
        phit  = np.empty(N, dtype=np.int8)
        bhit  = np.empty(N, dtype=np.int8)
        poses = [None]*N
        dims3 = np.empty((N,3), dtype=np.float32)
        tloc  = np.empty((N,3), dtype=np.float32)
        rloc6 = np.empty((N,6), dtype=np.float32)
        sigs  = [None]*N

        for j, c in enumerate(rows):
            qw,qx,qy,qz  = c[1][3:7]
            quats[j,:]   = (qw,qx,qy,qz)
            succ[j]      = 1 if c[2] else 0
            coll[j]      = 1 if c[3] else 0
            ghit[j]      = 1 if c[4] else 0
            phit[j]      = 1 if c[5] else 0
            bhit[j]      = 1 if c[6] else 0
            poses[j]     = c[1]
            ex           = c[7]
            faces[j]     = int(ex["final_face_id"])
            dims3[j,:]   = np.asarray(ex["dims"], dtype=np.float32)
            tloc[j,:]    = np.asarray(ex["t_loc"], dtype=np.float32)
            rloc6[j,:]   = np.asarray(ex["R_loc6"], dtype=np.float32)
            sigs[j]      = grasp_signature(ex["t_loc"], ex["R_loc6"])

        quats = quat_normalize(quats)

        groups = defaultdict(list)
        for idx, key in enumerate(sigs):
            groups[key].append(idx)

        for _, g_idx in groups.items():
            g_idx = np.asarray(g_idx, dtype=np.int64)

            clean_mask = (succ[g_idx]==1) & (coll[g_idx]==0) & (ghit[g_idx]==0) & (phit[g_idx]==0) & (bhit[g_idx]==0)
            clean_ids  = g_idx[clean_mask]
            if clean_ids.size == 0:
                continue

            g_faces = faces[g_idx]
            g_quats = quats[g_idx]
            g_succ  = succ[g_idx]
            g_coll  = coll[g_idx]
            g_poses = [poses[int(k)] for k in g_idx]
            g_dims  = dims3[g_idx]
            g_tloc  = tloc[g_idx]
            g_rloc6 = rloc6[g_idx]

            for li, gi in enumerate(clean_ids):
                pick_local = int(np.where(g_idx == gi)[0][0])

                fi = int(g_faces[pick_local])

                all_local = np.arange(g_idx.size, dtype=np.int64)
                not_self  = all_local != pick_local

                same_mask = (g_faces == fi)
                opp_mask  = (g_faces == _OPPOSITE[fi])
                adj_mask  = (~same_mask) & (~opp_mask)

                ang_deg = quat_angle_deg_abs(g_quats[pick_local], g_quats)
                small_mask  = same_mask & (ang_deg <= small_thresh_deg)
                medium_mask = same_mask & (~small_mask)

                bucket_lists = {
                    "SMALL":    all_local[small_mask & not_self],
                    "MEDIUM":   all_local[medium_mask & not_self],
                    "ADJACENT": all_local[adj_mask   & not_self],
                    "OPPOSITE": all_local[opp_mask   & not_self],
                }

                # target-driven allocation by BUCKET_TARGET (no binding caps)
                alloc = _compute_target_alloc(bucket_lists, group_cap)

                pick_t = g_tloc[pick_local].tolist()
                pick_R = g_rloc6[pick_local].tolist()

                for b in ("SMALL","MEDIUM","ADJACENT","OPPOSITE"):
                    k = alloc[b]
                    if k <= 0:
                        continue
                    idx_bucket = bucket_lists[b]
                    if idx_bucket.size == 0:
                        continue

                    target_succ_b = TARGET_SUCCESS_RATE_BY_BUCKET[b]
                    chosen_loc = sample_bucket_indices(idx_bucket, g_succ, g_coll, k, target_succ_b)
                    if chosen_loc.size == 0:
                        continue

                    init_pose_pick = g_poses[pick_local]

                    for j_local in chosen_loc.tolist():
                        pose_j = g_poses[j_local]
                        dims_j = g_dims[j_local].tolist()
                        face_j = int(g_faces[j_local])
                        s_j    = float(g_succ[j_local])
                        c_j    = float(g_coll[j_local])

                        obj = {
                            "t_loc": pick_t,
                            "R_loc6": pick_R,
                            "init_object_pose":  init_pose_pick,
                            "final_object_pose": pose_j,
                            "final_face_id": face_j,
                            "object_dimensions": dims_j,
                            "success_label": s_j,
                            "collision_label": c_j
                        }

                        key = _dims_key(obj["object_dimensions"])
                        tgt = pick_split_by_key(key)
                        if not first[tgt]:
                            outs[tgt].write(",\n")
                        else:
                            first[tgt] = False
                        outs[tgt].write(json.dumps(obj))

                        global_counts[b] += 1
                        total_pairs += 1
                        if (s_j > 0.5) and (c_j < 0.5):
                            succ_pairs += 1

    for k, f in outs.items():
        f.write("\n]"); f.close()

    if total_pairs > 0:
        succ_rate = succ_pairs / total_pairs
        print(f"[direct] GLOBAL: pairs={total_pairs}  success_rate={succ_rate:.3f}  (target={TARGET_SUCCESS_RATE:.3f})")
        for k in ("SMALL","MEDIUM","ADJACENT","OPPOSITE"):
            n = global_counts[k]
            print(f"         {k:<8} {n:>8}  {n/total_pairs:6.3f}")

    train_stats = _count_split_file(train_p)
    val_stats   = _count_split_file(val_p)
    test_stats  = _count_split_file(test_p)

    print(f"train.json: n={train_stats['n']}  success_rate={train_stats['succ_rate']:.3f}  collision_rate={train_stats['coll_rate']:.3f}")
    print(f"val.json:   n={val_stats['n']}    success_rate={val_stats['succ_rate']:.3f}    collision_rate={val_stats['coll_rate']:.3f}")
    print(f"test.json:  n={test_stats['n']}   success_rate={test_stats['succ_rate']:.3f}   collision_rate={test_stats['coll_rate']:.3f}")

# ============================ 4) SANITY (light) ============================

def run_sanity_checks(data_dir: str = DATA_DIR,
                      processed_subdir: str = PROCESSED_SUBDIR,
                      combined_subdir: str = COMBINED_SUBDIR,
                      sample_files: int = 3,
                      sample_rows: int = 200):
    proc_dir = os.path.join(data_dir, processed_subdir)
    files = sorted(glob.glob(os.path.join(proc_dir, "processed_object_*.json")))[:sample_files]
    print(f"[sanity] checking {len(files)} processed files")
    for fp in files:
        d = json.load(open(fp, "r"))
        n = len(d["t_loc_list"])
        assert n == len(d["R_loc6_list"]) == len(d["init_object_poses"]) == len(d["final_object_poses"]) \
            == len(d["final_face_ids"]) == len(d["object_dimensions"]) == len(d["success_labels"]) \
            == len(d["collision_labels"])
        print(f"  {os.path.basename(fp)}: n={n}")

    comb_dir = os.path.join(data_dir, combined_subdir)
    all_path = os.path.join(comb_dir, "all_data.json")
    print(f"[sanity] sampling {sample_rows} rows from {all_path}")

    pos = 0; n = 0; col = 0
    for obj in _stream_array(all_path):
        if obj["collision_label"] > 0.5: col += 1
        if (obj["success_label"] > 0.5) and (obj["collision_label"] < 0.5): pos += 1
        n += 1
        if n >= sample_rows: break
    print(f"[sanity] first {n} rows → success {pos/n:.3f} | collision {col/n:.3f}")
    print("[sanity] OK")

# ============================ main ============================

if __name__ == "__main__":
    # Direct streaming path: raw_data -> train/val/test (no intermediates)
    direct_split_from_raw()
