#!/usr/bin/env python3
"""
Build baseline trials from:
  - grasps.json  (object-frame grasps with meta: face, u_frac, v_frac, etc.)
  - test_deck_sim_10k.waypoints.jsonl  (10k pick/place pose pairs)

For each sampled task (Ti -> Tt), choose K grasps PER FACE (default 5) from faces {+X,-X,+Y,-Y},
and write one JSON line containing Ti, Tt, and the selected grasps.

Usage:
  python make_baseline_trials.py \
    --grasps /path/to/grasps.json \
    --traj   /path/to/test_deck_sim_10k.waypoints.jsonl \
    --out    /path/to/baseline_trials_900.jsonl \
    --n 900 --per-face 5 --seed 2025
"""

import argparse, json, random, os
from collections import defaultdict
from typing import Dict, List, Any

FACES_DEFAULT = ["+X","-X","+Y","-Y"]

def load_grasps(path: str, faces: List[str]) -> Dict[str, List[Dict[str,Any]]]:
    """
    Load grasps.json (dict[str->obj]) and group by 'face'.
    Sort each face bucket by centeredness |u-0.5| + |v-0.5|.
    """
    with open(path, "r") as f:
        data = json.load(f)

    groups: Dict[str, List[Dict[str,Any]]] = defaultdict(list)
    for gid, g in data.items():
        face = g.get("face")
        u = g.get("u_frac"); v = g.get("v_frac")
        if face not in faces or u is None or v is None:
            continue
        g_out = dict(g)  # shallow copy
        g_out["id"] = gid
        g_out["_centeredness"] = abs(float(u)-0.5) + abs(float(v)-0.5)
        groups[face].append(g_out)

    # sort each face bucket by centeredness (most centered first), then by id for stability
    for f in faces:
        groups[f].sort(key=lambda d: (d["_centeredness"], d["id"]))
    return groups

def sample_per_face(groups: Dict[str, List[Dict[str,Any]]], faces: List[str], k_per_face: int, rng: random.Random) -> List[Dict[str,Any]]:
    """
    For each face, randomly sample k_per_face grasps from the most centered subset.
    Strategy: take the top max(3*k_per_face, k_per_face) by centeredness as a candidate pool per face,
    then random.sample k_per_face (without replacement) for variety but still near center.
    """
    selected = []
    for f in faces:
        bucket = groups.get(f, [])
        top_pool = bucket[:max(3*k_per_face, k_per_face)]
        chosen = rng.sample(top_pool, k_per_face)
        # keep only the fields we actually need in the trials file
        for g in chosen:
            sel = {
                "id": g["id"],
                "face": g.get("face"),
                "u_frac": g.get("u_frac"),
                "v_frac": g.get("v_frac"),
                "position": g.get("position"),  # object-frame contact position
                "orientation_wxyz": g.get("orientation_wxyz"),  # object-frame tool orientation (wxyz)
                "axis_obj": g.get("axis_obj"),
                "angles_deg": g.get("angles_deg"),
                "dims_xyz": g.get("dims_xyz"),
                "version": g.get("version"),
            }
            selected.append(sel)
    return selected

def load_task_pairs_jsonl(path: str, n: int, rng: random.Random) -> List[Dict[str,Any]]:
    """
    Sample tasks uniformly across bins defined by (pick.p, place.p) where p in {0..9}.
    Extract Ti from 'pick_object_world' and Tt from 'place_object_world' (with fallbacks).
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Bin tasks by (pick.p, place.p)
    bins: Dict[tuple, List[Dict[str,Any]]] = defaultdict(list)
    for idx in range(len(lines)):
        rec = json.loads(lines[idx])

        # Extract binner keys (pick/place indices 0..9); skip if missing
        pick_obj = rec.get("pick") or {}
        place_obj = rec.get("place") or {}
        p_pick = pick_obj.get("p")
        p_place = place_obj.get("p")
        if p_pick is None or p_place is None:
            continue

        # Extract poses (keep full sub-objects so downstream code has R/t or q/t as needed)
        Ti = rec.get("pick_object_world") or rec.get("objW_pick") or rec.get("Ti") or rec.get("start")
        Tt = rec.get("place_object_world") or rec.get("objW_place") or rec.get("Tt") or rec.get("goal")
        if Ti is None or Tt is None:
            continue

        trial_id = rec.get("id", idx)
        bins[(int(p_pick), int(p_place))].append({"id": trial_id, "Ti": Ti, "Tt": Tt})

    # Create a deterministic ordering of non-empty bins
    non_empty_bins = sorted(list(bins.keys()))
    if not non_empty_bins:
        return []

    # Target per-bin quota and first pass selection
    per_bin = n // len(non_empty_bins)
    selected: List[Dict[str,Any]] = []
    leftovers: Dict[tuple, int] = {}

    for b in non_empty_bins:
        bucket = list(bins[b])
        rng.shuffle(bucket)
        take = min(per_bin, len(bucket))
        selected.extend(bucket[:take])
        leftovers[b] = take  # number already taken; acts as start index for leftovers
        bins[b] = bucket  # store shuffled order back

    # Distribute remainder in round-robin fashion over bins with remaining items
    remaining = n - len(selected)
    if remaining > 0:
        while remaining > 0:
            progressed = False
            for b in non_empty_bins:
                if remaining <= 0:
                    break
                start_idx = leftovers.get(b, 0)
                if start_idx < len(bins[b]):
                    selected.append(bins[b][start_idx])
                    leftovers[b] = start_idx + 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break  # no more items available across bins

    return selected[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int, default=900, help="Number of tasks to sample")
    ap.add_argument("--per-face", type=int, default=5, help="Number of grasps per face to include")
    ap.add_argument("--faces", nargs="+", default=FACES_DEFAULT, help="Faces to use (default: +X -X +Y -Y)")
    ap.add_argument("--seed",   type=int, default=2025, help="RNG seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load and group grasps by face using provided meta (face, u_frac, v_frac)
    GRASPS_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
    groups = load_grasps(GRASPS_PATH, faces=args.faces)
    print(f"Loaded {len(groups)} grasps by face")

    # Sample tasks from the 10k JSONL
    TRAJ_PATH = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.waypoints.jsonl"
    tasks = load_task_pairs_jsonl(TRAJ_PATH, n=args.n, rng=rng)
    print(f"Loaded {len(tasks)} tasks")
    # Write trials
    OUT_PATH = "/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_testing/baseline_trials.jsonl"
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as fout:
        for t in tasks:
            # Random (but centered)  selections per face for this task
            grasps_sel = sample_per_face(groups, faces=args.faces, k_per_face=args.per_face, rng=rng)

            record = {
                "trial_id": t["id"],
                "Ti": t["Ti"],           # full pick_object_world sub-struct
                "Tt": t["Tt"],           # full place_object_world sub-struct
                "policy": "Baseline",
                "faces": args.faces,
                "per_face": args.per_face,
                "grasp_ids": [g["id"] for g in grasps_sel],
                "grasps_obj": grasps_sel
            }
            fout.write(json.dumps(record) + "\n")

    print(f"[OK] Wrote {args.n} baseline trials to: {OUT_PATH}")

if __name__ == "__main__":
    main()
