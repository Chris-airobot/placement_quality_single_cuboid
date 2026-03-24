# build_waypoints_from_pairs.py
# Creates a waypoints deck ready for execution:
#   P1 (pick.pregrasp_world)
#   C1 (pick.grasp_pose_contact_world)
#   L1 (pick.lift_world)
#   L2 (place.lift_world)
#   P2 (place.pregrasp_world)
#   C2 (place.grasp_pose_contact_world)
#
# Also embeds pedestal poses (pick/place) from pedestal_poses.json so the
# runner doesn't need to look anywhere else.

import os, json

# ----- EDIT THESE PATHS IF NEEDED -------------------------------------------
RAW_ROOT            = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
IN_PATH             = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.jsonl"
OUT_PATH            = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.waypoints.jsonl"
PEDESTAL_POSES_PATH = "/home/chris/Chris/placement_ws/src/pedestal_poses.json"
# ----------------------------------------------------------------------------

def _ensure_dir_for(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _iter_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _load_endpoint_row(raw_root, p, g, o):
    """Open RAW_ROOT/p{p}/data_{g}.json and return row[str(o)]."""
    path = os.path.join(raw_root, f"p{int(p)}", f"data_{int(g)}.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data[str(int(o))]

def _to_wxyz(quat_like):
    """
    Convert a pedestal 'orientation' to WXYZ if needed.
    - If already a dict with 'orientation_quat', assume WXYZ and return.
    - If it's a list of 4 numbers (likely XYZW), convert to WXYZ.
    """
    if isinstance(quat_like, dict) and "orientation_quat" in quat_like:
        return list(map(float, quat_like["orientation_quat"]))
    q = list(map(float, quat_like))
    if len(q) == 4:
        # assume XYZW -> WXYZ
        return [q[3], q[0], q[1], q[2]]
    # fallback: return as-is
    return q

def _load_pedestal_poses(path):
    """
    Load pedestal poses. Expect a list of entries, each like:
      { "id": "...", "position":[x,y,z], "orientation":[x,y,z,w] }
    Returns a list indexed by pedestal index, with dicts:
      { "position":[...], "orientation_quat":[W,X,Y,Z] }
    """
    obj = _load_json(path)
    if isinstance(obj, list):
        src = obj
    elif isinstance(obj, dict) and "pedestals" in obj and isinstance(obj["pedestals"], list):
        src = obj["pedestals"]
    else:
        # minimal assumption: wrap single entry as list
        src = [obj]

    out = []
    for ent in src:
        pos = ent.get("position", [0,0,0])
        ori = ent.get("orientation_quat", ent.get("orientation", [0,0,0,1]))
        out.append({
            "position": list(map(float, pos)),
            "orientation_quat": _to_wxyz(ori),
        })
    return out

def _ped_pose_for_index(ped_poses, idx):
    ent = ped_poses[int(idx)]
    return {
        "position": ent["position"],
        "orientation_quat": ent["orientation_quat"],
    }

def main():
    _ensure_dir_for(OUT_PATH)
    ped_poses = _load_pedestal_poses(PEDESTAL_POSES_PATH)

    n = 0
    with open(OUT_PATH, "w") as fout:
        for rec in _iter_jsonl(IN_PATH):
            # pull pick/place keys
            p_pick  = rec["pick"]["p"]
            g_pick  = rec["pick"]["g"]
            o_pick  = rec["pick"]["o"]
            p_place = rec["place"]["p"]
            g_place = rec["place"]["g"]
            o_place = rec["place"]["o"]

            # load raw rows
            row_pick  = _load_endpoint_row(RAW_ROOT, p_pick,  g_pick,  o_pick)
            row_place = _load_endpoint_row(RAW_ROOT, p_place, g_place, o_place)

            # assemble six waypoints in the requested order (all world poses)
            waypoints = [
                {"name": "P1", "pose": row_pick["pregrasp_world"]},
                {"name": "C1", "pose": row_pick["grasp_pose_contact_world"]},
                {"name": "L1", "pose": row_pick["lift_world"]},
                {"name": "L2", "pose": row_place["lift_world"]},
                {"name": "P2", "pose": row_place["pregrasp_world"]},
                {"name": "C2", "pose": row_place["grasp_pose_contact_world"]},
            ]

            # object poses at pick/place (world)
            pick_obj  = row_pick.get("object_pose_world")
            place_obj = row_place.get("object_pose_world")

            # embed pedestal poses (indexed by p) into each record
            ped_pick_pose  = _ped_pose_for_index(ped_poses, p_pick)
            ped_place_pose = _ped_pose_for_index(ped_poses, p_place)

            out = {
                "id": rec.get("id"),
                "pick": rec["pick"],
                "place": rec["place"],
                "labels": rec.get("labels", {}),
                "bins": rec.get("bins", {}),
                "grasp": rec.get("grasp", {}),
                "comfort": rec.get("comfort", None),

                # six execution waypoints (world)
                "waypoints": waypoints,

                # world object poses at pick/place endpoints
                "pick_object_world":  pick_obj,
                "place_object_world": place_obj,

                # embed pedestal poses so the runner doesn't need to open pedestal_poses.json
                "pedestal": {
                    "pick_idx":  int(p_pick),
                    "place_idx": int(p_place),
                    "pick":  ped_pick_pose,   # {position: [...], orientation_quat: [W,X,Y,Z]}
                    "place": ped_place_pose,  # {position: [...], orientation_quat: [W,X,Y,Z]}
                },
            }

            fout.write(json.dumps(out) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"[build] wrote {n} records...")

    print(f"[build] done. wrote {n} waypoint records to {OUT_PATH}")

if __name__ == "__main__":
    main()
