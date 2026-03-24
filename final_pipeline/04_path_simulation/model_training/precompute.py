import os, json
from collections import OrderedDict
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
from placement_quality.path_simulation.model_training.model import (
    wxyz_to_R,
    rot_to_6d,
    corners_world,
    norm_pos_world,
    face_one_hot,
    BOX_DIMS,
    CORNERS_NORM,
)


# ---- paths & toggles (align with train.py/model.py) ----
RAW_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
PAIRS_DIR        = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs"
PAIRS_TRAIN      = os.path.join(PAIRS_DIR, "pairs_train.jsonl")
PAIRS_VAL        = os.path.join(PAIRS_DIR, "pairs_val.jsonl")
PAIRS_TEST       = os.path.join(PAIRS_DIR, "pairs_test.jsonl")

# Output directory root (will be specialized by toggles below)
PRECOMP_BASE     = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/precomputed"

# Feature toggles (must match what you want to train with)
USE_CORNERS = True
USE_META    = True
USE_DELTA   = True

# Ablation toggle (transport only; rotations fixed to 6D)
USE_TRANSPORT = False    # express grasp in transport frame

def _compose_precomp_root(base):
    # Write under a case-specific subfolder for clarity
    sub = []
    if USE_META:
        sub.append("meta")
    else:
        sub.append("nometa")
    sub.append("delta" if USE_DELTA else "abs")
    sub.append("corners" if USE_CORNERS else "nocorners")
    sub.append("transport" if USE_TRANSPORT else "world")
    return os.path.join(base, "-".join(sub))

PRECOMP_ROOT = _compose_precomp_root(PRECOMP_BASE)

# Memmap write chunking
LOG_EVERY = 200000


def _file(raw_root, p, g):
    return os.path.join(raw_root, f"p{int(p)}", f"data_{int(g)}.json")


def _load_row(cache, raw_root, p, g, o, max_cache):
    key = (int(p), int(g))
    if key in cache:
        cache.move_to_end(key, last=True)
    else:
        with open(_file(raw_root, p, g), "r") as f:
            data = json.load(f)
        cache[key] = data
        if len(cache) > max_cache:
            cache.popitem(last=False)
    return cache[key][str(int(o))]


def _count_lines(path):
    n = 0
    with open(path, "r") as f:
        for _ in f:
            n += 1
    return n


def _maybe_load_grasp_meta(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _alloc_memmaps(out_dir, n_samples, use_meta, use_corners):
    os.makedirs(out_dir, exist_ok=True)
    mm = {}
    d_pose = 9
    mm["grasp_Oi"]   = open_memmap(os.path.join(out_dir, "grasp_Oi.npy"), dtype=np.float32, mode="w+", shape=(n_samples, d_pose))
    mm["objW_pick"]  = open_memmap(os.path.join(out_dir, "objW_pick.npy"), dtype=np.float32, mode="w+", shape=(n_samples, d_pose))
    mm["objW_place"] = open_memmap(os.path.join(out_dir, "objW_place.npy"), dtype=np.float32, mode="w+", shape=(n_samples, d_pose))
    if use_meta:
        mm["meta"] = open_memmap(os.path.join(out_dir, "meta.npy"), dtype=np.float32, mode="w+", shape=(n_samples, 6))
    if use_corners:
        mm["corners_f"] = open_memmap(os.path.join(out_dir, "corners_f.npy"), dtype=np.float32, mode="w+", shape=(n_samples, 24))
    mm["y_ik"]  = open_memmap(os.path.join(out_dir, "y_ik.npy"), dtype=np.uint8, mode="w+", shape=(n_samples, 1))
    mm["y_col"] = open_memmap(os.path.join(out_dir, "y_col.npy"), dtype=np.uint8, mode="w+", shape=(n_samples, 1))
    return mm


def _finalize_memmaps(mm):
    # Flush to disk by deleting references
    for k in list(mm.keys()):
        mm[k].flush()
        del mm[k]


def _compute_corners_stats(pairs_path, raw_root, max_cache=8192):
    n = 0
    mean = np.zeros(24, dtype=np.float32)
    M2 = np.zeros(24, dtype=np.float32)
    cache = OrderedDict()
    with open(pairs_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            p2, g2, o2 = rec["place"]["p"], rec["place"]["g"], rec["place"]["o"]
            place = _load_row(cache, raw_root, p2, g2, o2, max_cache)
            pos_f  = np.array(place["object_pose_world"]["position"], dtype=np.float32)
            quat_f = np.array(place["object_pose_world"]["orientation_quat"], dtype=np.float32)
            R_WOf  = wxyz_to_R(quat_f)
            cf = corners_world(pos_f, R_WOf, BOX_DIMS)
            n += 1
            delta = cf - mean
            mean = mean + delta / float(n)
            delta2 = cf - mean
            M2 = M2 + delta * delta2
            if len(cache) > max_cache:
                cache.popitem(last=False)
    var = (M2 / max(1, n - 1)).astype(np.float32)
    std = np.sqrt(var) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def precompute_split(name, pairs_path, out_root, raw_root, grasps_meta_path, use_meta, use_corners, use_delta, max_cache=512):
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    total = _count_lines(pairs_path)
    mm = _alloc_memmaps(out_dir, total, use_meta, use_corners)

    gmeta = _maybe_load_grasp_meta(grasps_meta_path) if use_meta else None

    corners_mean = None
    corners_std = None
    if use_corners and CORNERS_NORM == "zscore":
        stats_mean_path = os.path.join(PRECOMP_ROOT, "corners_mean.npy")
        stats_std_path  = os.path.join(PRECOMP_ROOT, "corners_std.npy")
        if name == "train":
            if not (os.path.exists(stats_mean_path) and os.path.exists(stats_std_path)):
                corners_mean, corners_std = _compute_corners_stats(pairs_path, raw_root, max_cache=max_cache)
                np.save(stats_mean_path, corners_mean)
                np.save(stats_std_path,  corners_std)
            else:
                corners_mean = np.load(stats_mean_path)
                corners_std  = np.load(stats_std_path)
        else:
            if not (os.path.exists(stats_mean_path) and os.path.exists(stats_std_path)):
                raise FileNotFoundError("Corners z-score stats not found; run train split precompute first.")
            corners_mean = np.load(stats_mean_path)
            corners_std  = np.load(stats_std_path)

    cache = OrderedDict()
    i = 0
    with open(pairs_path, "r") as f:
        for line in tqdm(f, total=total, desc=f"precompute {name}"):
            rec = json.loads(line)
            p1, g,  o1 = rec["pick"]["p"],  rec["pick"]["g"],  rec["pick"]["o"]
            p2, g2, o2 = rec["place"]["p"], rec["place"]["g"], rec["place"]["o"]

            pick  = _load_row(cache, raw_root, p1, g,  o1, max_cache)
            place = _load_row(cache, raw_root, p2, g2, o2, max_cache)

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

            grasp_Oi   = np.concatenate([t_for_grasp, rot_to_6d(R_grasp)], axis=0)
            objW_pick  = np.concatenate([norm_pos_world(pos_i), rot_to_6d(R_WOi)], axis=0)
            if use_delta:
                t_delta = norm_pos_world(pos_f - pos_i)
                place_feat = np.concatenate([t_delta, rot_to_6d(R_delta)], axis=0)
            else:
                place_feat = np.concatenate([norm_pos_world(pos_f), rot_to_6d(R_WOf)], axis=0)

            mm["grasp_Oi"][i, :]   = grasp_Oi
            mm["objW_pick"][i, :]  = objW_pick
            mm["objW_place"][i, :] = place_feat

            if use_meta:
                gm = gmeta[int(g)]
                meta = np.concatenate([
                    np.array([float(gm.get("u_frac", 0.5)), float(gm.get("v_frac", 0.5))], dtype=np.float32),
                    face_one_hot(gm.get("face", None))
                ], axis=0)
                mm["meta"][i, :] = meta

            if use_corners:
                cf = corners_world(pos_f, R_WOf, BOX_DIMS)
                if CORNERS_NORM == "zscore" and corners_mean is not None and corners_std is not None:
                    cf = (cf - corners_mean) / corners_std
                mm["corners_f"][i, :] = cf.astype(np.float32)

            mm["y_ik"][i]  = np.uint8(rec["labels"]["ik"]) 
            mm["y_col"][i] = np.uint8(rec["labels"]["collision"]) 

            i += 1
            # progress handled by tqdm
            if len(cache) > max_cache:
                cache.popitem(last=False)

    _finalize_memmaps(mm)

    # Infer dims without touching closed memmaps
    d_pose = 9

    meta = {
        "n": total,
        "use_meta": use_meta,
        "use_corners": use_corners,
        "use_delta": use_delta,
        "rot_repr": "6d",
        "use_transport": USE_TRANSPORT,
        "dims": {"grasp_Oi": d_pose, "objW_pick": d_pose, "objW_place": d_pose, "meta": 6 if use_meta else 0, "corners_f": 24 if use_corners else 0},
        "corners_norm": CORNERS_NORM,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] wrote precomputed arrays to: {out_dir}")


def main():
    os.makedirs(PRECOMP_ROOT, exist_ok=True)
    print("Starting precomputation…")
    precompute_split("train", PAIRS_TRAIN, PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)
    precompute_split("val",   PAIRS_VAL,   PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)
    precompute_split("test",  PAIRS_TEST,  PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)
    print("Done.")


if __name__ == "__main__":
    main()


