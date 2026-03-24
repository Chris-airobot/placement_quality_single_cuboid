# post_processing.py  — edge-based splits (routes) instead of pedestal holdout
import os, json, glob, math, random
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ====== PATHS (set these) ===================================================
RAW_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
OUTPUT_DIR       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs"
PAIRS_TRAIN_PATH = os.path.join(OUTPUT_DIR, "pairs_train.jsonl")  # existing train pairs

# Rebuild only val & test using stratified edges (keep train fixed):
REBUILD_VAL_TEST_ONLY = False
# ============================================================================

# ----- toggles / knobs -------------------------------------------------------
RUN_SANITY_ONLY = False        # True = run only sanity & dry-run checks; False = also write pairs
K_TRAIN, K_EVAL = 24, 32       # K-per-pick (train / val&test)
R_MAX = 48                     # max times a place endpoint can be reused
RNG_SEED = 42
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
CLEARANCE_MIN = 0.005          # 5 mm threshold for pregrasp clearance sanity (NOT used for filtering)
EDGE_SPLIT = (0.80, 0.10, 0.10)  # train/val/test fractions over directed edges (i->j), i and j can be equal
# New split constraints
KEEP_TRAIN_FIXED = False          # if True, keep existing train edges fixed; else rebuild train/val/test
ENFORCE_SOURCE_COVERAGE = True    # ensure every pedestal appears as a pick source in each split where possible

# ----- SAMPLE MODE (fast iteration) ------------------------------------------
SAMPLE_MODE = False              # set True for quick, small runs
PED_SAMPLE = [0, 1, 2,3,4,5]          # subset of pedestals to include
MAX_FILES_PER_PED = 50           # limit number of grasp files per pedestal
MAX_O_PER_FILE = 120             # limit number of orientation keys per file
MAX_PAIRS_PER_SPLIT = 20000     # stop writing pairs for a split when reached (sample mode)
OUTPUT_DIR_SAMPLE_SUFFIX = "_sample"

def _resolve_out_path(path):
    if not SAMPLE_MODE:
        return path
    out_dir, out_base = os.path.split(path)
    out_dir = out_dir + OUTPUT_DIR_SAMPLE_SUFFIX
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, out_base)

# ----- helpers ---------------------------------------------------------------
def _wxyz_to_R(qwxyz):
    q = np.array(qwxyz, dtype=float)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def _quat_angle_deg(q1_wxyz, q2_wxyz):
    a = np.array(q1_wxyz, float); a /= (np.linalg.norm(a) + 1e-12)
    b = np.array(q2_wxyz, float); b /= (np.linalg.norm(b) + 1e-12)
    dot = abs(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))

# --- NEW: fast helpers -------------------------------------------------------
def _normalize_quat_wxyz(qwxyz):
    q = np.array(qwxyz, float)
    n = float(np.linalg.norm(q)) + 1e-12
    return (q / n).tolist()

def _quat_angle_deg_unit(q1_wxyz_unit, q2_wxyz_unit):
    # assumes inputs already unit-normalized
    dot = abs(np.clip(float(np.dot(q1_wxyz_unit, q2_wxyz_unit)), -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))


def _bottom_face_label(R_WO):
    faces = {
        "+X": np.array([ 1,0,0], float), "-X": np.array([-1,0,0], float),
        "+Y": np.array([ 0,1,0], float), "-Y": np.array([ 0,-1,0], float),
        "+Z": np.array([ 0,0,1], float), "-Z": np.array([ 0,0,-1], float),
    }
    dots = {f: float((R_WO @ n).dot(WORLD_UP)) for f, n in faces.items()}
    return min(dots, key=dots.get)

def _dist_bin(p1_xy, p2_xy, thr12):
    if np.allclose(p1_xy, p2_xy): return 0
    d = float(np.linalg.norm(np.asarray(p1_xy) - np.asarray(p2_xy)))
    t1, t2 = thr12
    return 1 if d <= t1 else (2 if d <= t2 else 3)

def _dori_bin(deg):
    if deg <= 1e-3: return 0
    return 1 if deg <= 36 else (2 if deg <= 90 else 3)

def _half_extent_along_world_up(R_WO, dims_xyz):
    proj = np.abs(R_WO.T @ WORLD_UP)  # components in object axes
    return 0.5 * float(np.sum(proj * np.asarray(dims_xyz, dtype=float)))

# ----- load meta -------------------------------------------------------------
def load_grasp_meta(path):
    with open(path, "r") as f:
        meta_raw = json.load(f)
    meta = {}
    global_dims = None
    for k, g in meta_raw.items():
        g_id = int(k)
        axis = np.array(g.get("axis_obj", [0,0,0]), float)
        dims = np.array(g.get("dims_xyz", [0,0,0]), float)
        if global_dims is None:
            global_dims = dims.copy()
        half = 0.5 * dims
        jaw_span = float(2.0 * np.sum(np.abs(axis) * half))
        meta[g_id] = {
            "face": g.get("face", None),
            "u_frac": g.get("u_frac", None),
            "v_frac": g.get("v_frac", None),
            "axis_obj": axis,
            "dims_xyz": dims,
            "jaw_span": jaw_span,
            "jaw_span_ok": (jaw_span <= 0.08),
        }
    return meta, (global_dims if global_dims is not None else np.array([0,0,0], float))

# ----- pedestals, centers, thresholds ---------------------------------------
def list_pedestals(raw_root):
    ps = []
    for d in sorted(glob.glob(os.path.join(raw_root, "p*"))):
        name = os.path.basename(d)
        if name.startswith("p") and name[1:].isdigit():
            ps.append(int(name[1:]))
    ps = sorted(ps)
    if SAMPLE_MODE:
        ps = [p for p in ps if p in set(PED_SAMPLE)]
    return ps

def pedestal_centers_xy(raw_root, ped_ids):
    centers = {}
    for p in ped_ids:
        folder = os.path.join(raw_root, f"p{p}")
        files = sorted(glob.glob(os.path.join(folder, "data_*.json")))
        if not files: continue
        with open(files[0], "r") as f:
            data = json.load(f)
        first_key = min(data.keys(), key=lambda x:int(x))
        xy = data[first_key]["object_pose_world"]["position"][:2]
        centers[p] = (float(xy[0]), float(xy[1]))
    return centers

def distance_thresholds(centers):
    vals = []
    ids = sorted(centers.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            d = np.linalg.norm(np.array(centers[ids[i]]) - np.array(centers[ids[j]]))
            if d > 1e-9: vals.append(d)
    vals = sorted(vals)
    if not vals: return (0.1, 0.2)
    q1 = vals[int(len(vals)*0.33)]
    q2 = vals[int(len(vals)*0.66)]
    return (q1, q2)

# ----- build ONE global index over all pedestals -----------------------------
def build_global_indices(raw_root, grasp_meta, global_dims):
    split_obj = {"grasp_to_eps": defaultdict(list), "picks": [], "all_eps": set(),
                 "counts": Counter(), "per_grasp_counts": Counter(),
                 "clearance_hist": []}

    ped_ids = list_pedestals(raw_root)
    for p in tqdm(ped_ids, desc="[scan] pedestals"):
        folder = os.path.join(raw_root, f"p{p}")
        files = sorted(glob.glob(os.path.join(folder, "data_*.json")))
        if SAMPLE_MODE:
            files = files[:max(0, int(MAX_FILES_PER_PED))]
        for fp in tqdm(files, leave=False, desc=f"  files p{p}"):
            g = int(os.path.splitext(os.path.basename(fp))[0].split("_")[1])
            jaw_ok = grasp_meta.get(g, {}).get("jaw_span_ok", False)
            face_g = grasp_meta.get(g, {}).get("face", None)
            with open(fp, "r") as f:
                data = json.load(f)
            seen_o = set()
            o_items = list(data.items())
            if SAMPLE_MODE:
                o_items = o_items[:max(0, int(MAX_O_PER_FILE))]
            for o_str, row in o_items:
                o = int(o_str); seen_o.add(o)
                ik = row["ik_endpoints"]
                ik_ok = bool(ik["C"]["ok"]) and bool(ik["P"]["ok"]) and bool(ik["L"]["ok"])
                seg = row["local_segments"]
                path_ok_pick  = bool(seg["P_to_C"]["ok"]) and bool(seg["C_to_L"]["ok"])
                path_ok_place = bool(seg["P_to_C"]["ok_env"]) and bool(seg["C_to_L"]["ok_env"])
                qwxyz_raw = row["object_pose_world"]["orientation_quat"]
                qwxyz = _normalize_quat_wxyz(qwxyz_raw)  # store unit quaternion
                R_WO = _wxyz_to_R(qwxyz)
                bottom = _bottom_face_label(R_WO)
                support_blocked = (face_g is not None and face_g == bottom)
                xy = tuple(row["object_pose_world"]["position"][:2])
                # clearance (NOT used for filtering)
                pre_z = float(row["pregrasp_world"]["position"][2])
                half_up = _half_extent_along_world_up(R_WO, global_dims)
                obj_z  = float(row["object_pose_world"]["position"][2])
                ped_top_z = obj_z - half_up
                clearance = pre_z - ped_top_z
                clear_ok = (clearance >= CLEARANCE_MIN)

                # NEW: collision-at-contact policy (two-stage)
                # - pick_col_free_any: pick endpoints must be free of ANY collision
                # - place_col_free_gp: place label ignores box-only collisions (only ground/pedestal count)
                col_dict = row.get("endpoint_collision_at_C", {})
                pick_col_free_any = not bool(col_dict.get("any", False))
                place_col_free_gp = not (bool(col_dict.get("ground", False)) or bool(col_dict.get("pedestal", False)))

                # NOTE: we store place_col_free_gp in the tuple's last slot (index 9),
                # replacing the old 'clear_ok' there (clear_ok is still counted above).
                ep = (p, g, o, ik_ok, path_ok_place, tuple(qwxyz), xy, jaw_ok, support_blocked, place_col_free_gp)

                split_obj["grasp_to_eps"][g].append(ep)
                split_obj["all_eps"].add((p,g,o))
                split_obj["per_grasp_counts"][g] += 1

                # Pick safety gate now REQUIRES collision-free against ANY contact:
                if ik_ok and path_ok_pick and jaw_ok and not support_blocked and pick_col_free_any:
                    split_obj["picks"].append(ep)


                c = split_obj["counts"]
                c["endpoints"] += 1
                c["ik_ok"] += int(ik_ok)
                c["path_ok"] += int(path_ok_pick)
                c["jaw_ok"] += int(jaw_ok)
                c["support_blocked"] += int(support_blocked)
                c["clear_ok"] += int(clear_ok)
                split_obj["clearance_hist"].append(clearance)

            split_obj["counts"]["files"] += 1
            split_obj["counts"]["unique_o_keys_all_120"] += int(len(seen_o) == 120)

    return split_obj

# ----- edge partitioning -----------------------------------------------------
def build_edge_splits_stratified_val_test(ped_ids, centers, thr12, seed=RNG_SEED, include_self=True,
                                          train_edges_fixed=None, frac_val_test=(0.5, 0.5)):
    """
    Keep train edges fixed (read from pairs_train.jsonl).
    Split the remaining edges (pool) into val/test, stratified by distance bins (near/mid/far/self).
    """
    random.seed(seed)

    # All directed edges
    all_edges = []
    for i in ped_ids:
        for j in ped_ids:
            if not include_self and i == j:
                continue
            all_edges.append((i, j))

    train_edges_fixed = train_edges_fixed or set()
    train_edges = set()
    val_edges = set()
    test_edges = set()

    if KEEP_TRAIN_FIXED:
        pool = [e for e in all_edges if e not in train_edges_fixed]
        train_edges = set(train_edges_fixed)
        # Bin pool edges by distance
        binned = {0: [], 1: [], 2: [], 3: []}
        for (i, j) in pool:
            db = _dist_bin(centers[i], centers[j], thr12)
            binned[db].append((i, j))
        # Split within each bin with ratio frac_val_test
        r = float(frac_val_test[0]) / max(1e-9, (frac_val_test[0] + frac_val_test[1]))
        for db in [0, 1, 2, 3]:
            edges_bin = binned[db]
            random.shuffle(edges_bin)
            n_val = int(len(edges_bin) * r)
            val_edges.update(edges_bin[:n_val])
            test_edges.update(edges_bin[n_val:])
    else:
        # Rebuild train/val/test from scratch by distance bins using EDGE_SPLIT
        binned = {0: [], 1: [], 2: [], 3: []}
        for (i, j) in all_edges:
            db = _dist_bin(centers[i], centers[j], thr12)
            binned[db].append((i, j))
        for db in [0, 1, 2, 3]:
            edges_bin = binned[db]
            random.shuffle(edges_bin)
            n_total = len(edges_bin)
            n_train = int(round(EDGE_SPLIT[0] * n_total))
            n_val   = int(round(EDGE_SPLIT[1] * n_total))
            n_test  = max(0, n_total - (n_train + n_val))
            train_edges.update(edges_bin[:n_train])
            val_edges.update(edges_bin[n_train:n_train+n_val])
            test_edges.update(edges_bin[n_train+n_val: n_train+n_val+n_test])

    # Enforce source (pick) coverage in each split if requested
    if ENFORCE_SOURCE_COVERAGE:
        def sources(S):
            return set(i for (i, _j) in S)
        # helper to move one edge with source p from donor to target
        def move_source(p, donor, target):
            for (ii, jj) in list(donor):
                if ii == p:
                    donor.remove((ii, jj))
                    target.add((ii, jj))
                    return True
            return False
        # iterate until no missing or no progress
        splits = [("train", train_edges), ("val", val_edges), ("test", test_edges)]
        progressed = True
        while progressed:
            progressed = False
            for name, S in splits:
                miss = [p for p in ped_ids if p not in sources(S)]
                if not miss:
                    continue
                for p in miss:
                    # try donors in priority order with largest set size to reduce ratio drift
                    donors = sorted([(n, D) for (n, D) in splits if D is not S], key=lambda x: -len(x[1]))
                    for _dn, D in donors:
                        if move_source(p, D, S):
                            progressed = True
                            break
                # continue outer loop to recompute miss for next split

    # Return dict; we also return train for reporting/dry-run if desired
    return {"train": set(train_edges), "val": set(val_edges), "test": set(test_edges)}

# ----- DRY-RUN pairing (no writes) with EDGE whitelist ----------------------
def dry_run_pairing_stats_edge(split_name, split_obj, edge_whitelist, ped_centers, thr12, K, R_cap):
    random.seed(RNG_SEED)
    picks = list(split_obj["picks"])
    g2eps = split_obj["grasp_to_eps"]

    used = Counter()
    bin_counts = Counter()
    pool_sizes = []
    per_pick_taken = []
    empty_pool = 0

    for (p,g,o,ik_ok,pt_ok,qwxyz,xy,jaw_ok,supp_blk,clear_ok) in tqdm(picks, desc=f"[dry] {split_name}", leave=False):
        # candidate pool filtered by: same grasp g, not the same endpoint, jaw_ok & not support_blocked,
        # AND edge (p -> p2) allowed by whitelist
        pool = [ep for ep in g2eps.get(g, [])
                if not (ep[0]==p and ep[1]==g and ep[2]==o)
                and ep[7] and not ep[8]
                and (p, ep[0]) in edge_whitelist]
        pool_sizes.append(len(pool))
        if not pool:
            empty_pool += 1
            per_pick_taken.append(0)
            continue

        bins = defaultdict(list)
        for ep in pool:
            p2,g2,o2,ik2,pt2,q2,xy2,jok2,sb2,col2 = ep  # col2 = place collision-free flag
            db = _dist_bin(ped_centers[p], ped_centers[p2], thr12)
            ab = _dori_bin(_quat_angle_deg_unit(qwxyz, q2))  # both are unit quats now
            pb = 1 if pt2 else 0
            bins[(db,ab,pb)].append(ep)


        taken = 0
        for db in [0,1,2,3]:
            for ab in [0,1,2,3]:
                for pb in [0,1]:
                    lst = bins.get((db,ab,pb), [])
                    if not lst: continue
                    lst.sort(key=lambda e: (used[(e[0],e[1],e[2])], random.random()))
                    cand = None
                    for e in lst:
                        key = (e[0],e[1],e[2])
                        if used[key] < R_cap:
                            cand = e; break
                    if cand is None: continue
                    used[(cand[0],cand[1],cand[2])] += 1
                    bin_counts[(db,ab,pb)] += 1
                    taken += 1
                    if taken >= K: break
                if taken >= K: break
            if taken >= K: break

        per_pick_taken.append(taken)

    use_vals   = np.array(list(used.values()), dtype=float) if used else np.zeros(0)
    pool_vals  = np.array(pool_sizes, dtype=float) if pool_sizes else np.zeros(0)
    taken_vals = np.array(per_pick_taken, dtype=float) if per_pick_taken else np.zeros(0)

    stats = {
        "n_picks": len(picks),
        "pool_empty_frac": float(empty_pool) / max(1, len(picks)),
        "pool_size": {
            "mean": float(pool_vals.mean()) if pool_vals.size else 0.0,
            "p10": float(np.percentile(pool_vals, 10)) if pool_vals.size else 0.0,
            "p50": float(np.percentile(pool_vals, 50)) if pool_vals.size else 0.0,
            "p90": float(np.percentile(pool_vals, 90)) if pool_vals.size else 0.0,
        },
        "per_pick_taken": {
            "mean": float(taken_vals.mean()) if taken_vals.size else 0.0,
            "fullK_frac": float(np.mean(taken_vals >= K)) if taken_vals.size else 0.0,
        },
        "place_usage": {
            "max": int(use_vals.max()) if use_vals.size else 0,
            "p95": float(np.percentile(use_vals, 95)) if use_vals.size else 0.0,
            "p99": float(np.percentile(use_vals, 99)) if use_vals.size else 0.0,
        },
        "bin_counts": {str(k): int(v) for k, v in bin_counts.items()},
        "bin_zero_frac": _bin_zero_fraction(bin_counts),
        "top_used_places": _top_used_places(used, n=10),
    }
    return stats

def _bin_zero_fraction(bin_counts):
    total_bins = 4*4*2  # dist(4)*dori(4)*place_ok(2)
    filled = len(bin_counts)
    return float(max(0, total_bins - filled)) / float(total_bins)

def _top_used_places(used_counter, n=10):
    if not used_counter: return []
    top = used_counter.most_common(n)
    return [{"p":k[0], "g":k[1], "o":k[2], "use":int(v)} for k, v in top]

def _load_train_edges_from_pairs(path):
    """
    Read existing pairs_train.jsonl and return the whitelist of directed edges (pick.p -> place.p).
    """
    E = set()
    if not os.path.exists(path):
        return E
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            E.add((int(rec["pick"]["p"]), int(rec["place"]["p"])))
    return E

# ----- actual pairing (writes) with EDGE whitelist --------------------------
def sample_pairs_for_split_edge(split_name, split_obj, edge_whitelist, ped_centers, thr12, K, R_cap, out_path):
    random.seed(RNG_SEED)
    picks = split_obj["picks"]
    g2eps = split_obj["grasp_to_eps"]
    used = Counter()
    n_pairs = 0
    # --- class distribution counters ---
    ik_pos = ik_neg = 0
    col_pos = col_neg = 0
    joint_counts = Counter()  # keys: (ik,col)

    # Output path: use a sample suffix if enabled
    out_dir, out_base = os.path.split(out_path)
    if SAMPLE_MODE:
        out_dir = out_dir + OUTPUT_DIR_SAMPLE_SUFFIX
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_base)

    with open(out_path, "w") as fout, tqdm(total=len(picks), desc=f"[pair] {split_name}") as pbar:
        for (p,g,o,ik_ok,pt_ok,qwxyz,xy,jaw_ok,supp_blk,pick_col_free) in picks:
            # candidate pool restricted by edge whitelist
            pool = [ep for ep in g2eps.get(g, [])
                    if not (ep[0]==p and ep[1]==g and ep[2]==o)
                    and ep[7] and not ep[8]
                    and (p, ep[0]) in edge_whitelist]
            if pool:
                bins = defaultdict(list)
                for ep in pool:
                    p2,g2,o2,ik2,pt2,q2,xy2,jok2,sb2,col2 = ep
                    db = _dist_bin(ped_centers[p], ped_centers[p2], thr12)
                    ab = _dori_bin(_quat_angle_deg_unit(qwxyz, q2))
                    pb = 1 if pt2 else 0
                    bins[(db,ab,pb)].append(ep)

                taken = 0
                for db in [0,1,2,3]:
                    for ab in [0,1,2,3]:
                        for pb in [0,1]:
                            lst = bins.get((db,ab,pb), [])
                            if not lst: continue
                            lst.sort(key=lambda e: (used[(e[0],e[1],e[2])], random.random()))
                            cand = None
                            for e in lst:
                                key = (e[0],e[1],e[2])
                                if used[key] < R_cap:
                                    cand = e; break
                            if cand is None: continue
                            used[(cand[0],cand[1],cand[2])] += 1
                            # Decoupled labels with new naming:
                            # - IK feasibility from place ik_ok (cand[3])
                            ik_lbl  = 1 if cand[3] else 0
                            # Path-aware place label:
                            # - If IK is feasible (cand[3]), require BOTH a clean endpoint (cand[9]) AND a clean sweep (cand[4]).
                            # - If IK is not feasible, sweeps are "unknown/not-run", so fall back to endpoint-only.
                            col_lbl = 1 if (cand[9] and (not cand[3] or cand[4])) else 0
                            # --- update class counters ---
                            if ik_lbl:  ik_pos += 1
                            else:       ik_neg += 1
                            if col_lbl: col_pos += 1
                            else:       col_neg += 1
                            joint_counts[(ik_lbl, col_lbl)] += 1
                            # Write both backward-compatible and new, clearer names
                            rec = {
                                "pick":  {"p":p,"g":g,"o":o},
                                "place": {"p":cand[0],"g":cand[1],"o":cand[2]},
                                "labels":{
                                    "ik": ik_lbl,
                                    "collision": col_lbl,
                                    "ik_feasible": ik_lbl,
                                    "col_free": col_lbl
                                },
                                "bins":  {"dist":db,"dori":ab,"place_ok":1 if cand[4] else 0}
                            }
                            fout.write(json.dumps(rec) + "\n")
                            n_pairs += 1; taken += 1
                            if taken >= K: break
                        if taken >= K: break
                    if taken >= K: break
            pbar.update(1)
            if SAMPLE_MODE and n_pairs >= int(MAX_PAIRS_PER_SPLIT):
                break

    print(f"[{split_name}] class distribution — "
          f"IK: pos={ik_pos:,} neg={ik_neg:,}  |  "
          f"Collision: pos={col_pos:,} neg={col_neg:,}  |  total_pairs={n_pairs:,}")
    # Joint buckets
    j11 = joint_counts.get((1,1), 0)
    j10 = joint_counts.get((1,0), 0)
    j01 = joint_counts.get((0,1), 0)
    j00 = joint_counts.get((0,0), 0)
    print(f"[{split_name}] joint buckets — (1,1)={j11:,} (1,0)={j10:,} (0,1)={j01:,} (0,0)={j00:,}")
    return n_pairs, used

# ----- sanity reports --------------------------------------------------------
def sanity_report_basic_global(split_obj):
    c = split_obj["counts"]
    n_ep = c["endpoints"]; n_pick = len(split_obj["picks"])
    print(f"[global] endpoints={n_ep}  picks={n_pick}  "
          f"ik_ok={c['ik_ok']}/{n_ep}  path_ok={c['path_ok']}/{n_ep}  "
          f"jaw_ok={c['jaw_ok']}/{n_ep}  support_blocked={c['support_blocked']}  "
          f"clear_ok={c['clear_ok']}/{n_ep}  "
          f"files={c['files']}  all_o_keys_120={c['unique_o_keys_all_120']}/{c['files']}")

def sanity_report_per_grasp(split_obj, tag):
    counts = split_obj["per_grasp_counts"]
    vals = np.array(list(counts.values()), dtype=float) if counts else np.zeros(0)
    if vals.size:
        print(f"[{tag}] per-grasp endpoints: mean={vals.mean():.1f}  p10={np.percentile(vals,10):.1f}  "
              f"p50={np.percentile(vals,50):.1f}  p90={np.percentile(vals,90):.1f}  max={vals.max():.1f}  (unique grasps={len(vals)})")
    else:
        print(f"[{tag}] per-grasp endpoints: no data")

def sanity_report_edges(edge_splits, ped_ids):
    def _cov(S):
        nodes = sorted(set([i for (i,_) in S] + [j for (_,j) in S]))
        return nodes
    def _sources(S):
        return sorted(set(i for (i, _j) in S))
    print("Edge split sizes:",
          {k: len(v) for k,v in edge_splits.items()})
    for name in ["train","val","test"]:
        nodes = _cov(edge_splits[name])
        print(f"[edges:{name}] node coverage {len(nodes)}/{len(ped_ids)} → {sorted(nodes)}")
        srcs = _sources(edge_splits[name])
        print(f"[edges:{name}] source (pick) coverage {len(srcs)}/{len(ped_ids)} → {srcs}")
    # disjointness already asserted at build time

def _bin_stats(path, max_lines=20000):
    bins = Counter(); cnt = 0
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line); b = rec["bins"]
            bins[(b["dist"], b["dori"], b["place_ok"])] += 1
            cnt += 1
            if cnt >= max_lines: break
    return cnt, bins

# ----- main -----------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # discover pedestals + geometry
    ped_ids = list_pedestals(RAW_ROOT)
    assert len(ped_ids) >= 3, "Need ≥3 pedestals"
    centers = pedestal_centers_xy(RAW_ROOT, ped_ids)
    thr12 = distance_thresholds(centers)

    # load grasp meta + build one global index
    grasp_meta, global_dims = load_grasp_meta(GRASPS_META_PATH)
    split_obj = build_global_indices(RAW_ROOT, grasp_meta, global_dims)

    # edges: split directed routes (i->j) into train/val/test
    train_edges_fixed = _load_train_edges_from_pairs(PAIRS_TRAIN_PATH)
    edge_splits = build_edge_splits_stratified_val_test(
        ped_ids, centers, thr12, seed=RNG_SEED, include_self=True,
        train_edges_fixed=train_edges_fixed, frac_val_test=(EDGE_SPLIT[1], EDGE_SPLIT[2])
    )

    # --- SANITY: global counts + integrity ---
    sanity_report_basic_global(split_obj)
    sanity_report_per_grasp(split_obj, "global")
    sanity_report_edges(edge_splits, ped_ids)

    # --- SANITY: dry-run pairing (same K/R as real), with edge whitelist ---
    names = [("val", K_EVAL), ("test", K_EVAL)] if REBUILD_VAL_TEST_ONLY else [("train", K_TRAIN), ("val", K_EVAL), ("test", K_EVAL)]
    for name, K in names:
        stats = dry_run_pairing_stats_edge(name, split_obj, edge_splits[name], centers, thr12, K, R_MAX)
        print(f"[dry:{name}] picks={stats['n_picks']:,}  pool_empty_frac={stats['pool_empty_frac']:.3f}")
        ps, tk, pu = stats["pool_size"], stats["per_pick_taken"], stats["place_usage"]
        print(f"[dry:{name}] pool_size mean={ps['mean']:.1f}  p10={ps['p10']:.1f}  p50={ps['p50']:.1f}  p90={ps['p90']:.1f}")
        print(f"[dry:{name}] taken mean={tk['mean']:.1f}  fullK_frac={tk['fullK_frac']:.3f} (target→≈1.0)")
        print(f"[dry:{name}] place_usage max={pu['max']}  p95={pu['p95']:.1f}  p99={pu['p99']:.1f}  (cap={R_MAX})")
        print(f"[dry:{name}] zero_bin_frac={stats['bin_zero_frac']:.3f} (0.0 means all 32 bins hit)")
        top = stats["top_used_places"]
        if top: print(f"[dry:{name}] top_used_places(sample): {top[:5]}")
        sample_bins = dict(list(stats["bin_counts"].items())[:16])
        print(f"[dry:{name}] bin_counts(sample): {sample_bins}")

    # if RUN_SANITY_ONLY:
    #     print("Sanity & dry-run checks complete. Set RUN_SANITY_ONLY=False to generate pairs.")
    #     return

    # --- Pairing (writes) with edge whitelist ---
    out_train = _resolve_out_path(os.path.join(OUTPUT_DIR, "pairs_train.jsonl"))
    out_val   = _resolve_out_path(os.path.join(OUTPUT_DIR, "pairs_val.jsonl"))
    out_test  = _resolve_out_path(os.path.join(OUTPUT_DIR, "pairs_test.jsonl"))

    n_train = 0
    if not REBUILD_VAL_TEST_ONLY:
        n_train, _ = sample_pairs_for_split_edge("train", split_obj, edge_splits["train"], centers, thr12, K_TRAIN, R_MAX, out_train)
    # Always rebuild val & test
    n_val,  _ = sample_pairs_for_split_edge("val",  split_obj, edge_splits["val"],  centers, thr12, K_EVAL, R_MAX, out_val)
    n_test, _ = sample_pairs_for_split_edge("test", split_obj, edge_splits["test"], centers, thr12, K_EVAL, R_MAX, out_test)

    print(f"Pairs written: train={n_train:,}, val={n_val:,}, test={n_test:,}")

    paths = [("val", out_val), ("test", out_test)] if REBUILD_VAL_TEST_ONLY else [("train", out_train), ("val", out_val), ("test", out_test)]
    for name, path in paths:
        cnt, bins = _bin_stats(path)
        print(f"[{name}] quick bin sample (first {cnt}): {dict(bins)}")

if __name__ == "__main__":
    main()
