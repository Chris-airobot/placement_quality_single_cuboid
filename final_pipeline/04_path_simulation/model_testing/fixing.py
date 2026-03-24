import os
import json
import argparse


DEFAULT_RAW_ROOT = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"


def load_raw_case(raw_root, pedestal_idx, grasp_id, cache):
    """
    Load and cache the raw data JSON for a given (pedestal, grasp).
    Returns the parsed dict.
    """
    key = (int(pedestal_idx), int(grasp_id))
    if key in cache:
        return cache[key]
    p_dir = os.path.join(raw_root, f"p{int(pedestal_idx)}")
    path = os.path.join(p_dir, f"data_{int(grasp_id)}.json")
    with open(path, "r") as f:
        data = json.load(f)
    cache[key] = data
    return data


def corrected_collision_free(raw_entry):
    """
    Given a raw entry (for a specific orientation key), determine if it should be
    considered collision-free under the updated policy:
    - ground or pedestal collision => collision (False)
    - box-only collision => collision-free (True)
    - no collision at all => collision-free (True)
    """
    col = raw_entry.get("endpoint_collision_at_C", {})
    ground = bool(col.get("ground", False))
    pedestal = bool(col.get("pedestal", False))
    # box flag present but does not contribute to collision per new policy
    return not (ground or pedestal)


def choose_label_key(labels_obj):
    """
    Support both 'collision' (pairs) and 'col' (waypoints) keys.
    Prefer whichever exists; if both exist, update both.
    Returns a tuple (has_collision, has_col).
    """
    return ("collision" in labels_obj, "col" in labels_obj)


def process(input_path, output_path, raw_root):
    raw_cache = {}
    total = 0
    changed = 0
    flips_to_one = 0
    flips_to_zero = 0
    box_only_cases = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            place = rec.get("place", {})
            p_idx = int(place.get("p"))
            g_id = int(place.get("g"))
            o_idx = int(place.get("o"))

            raw = load_raw_case(raw_root, p_idx, g_id, raw_cache)
            raw_entry = raw[str(o_idx)]

            col_free = corrected_collision_free(raw_entry)
            lbl_val = 1 if col_free else 0

            # Count box-only for reporting
            col_dict = raw_entry.get("endpoint_collision_at_C", {})
            if (not bool(col_dict.get("ground", False))) and (not bool(col_dict.get("pedestal", False))) and bool(col_dict.get("box", False)):
                box_only_cases += 1

            labels = rec.get("labels", {})
            has_collision_key, has_col_key = choose_label_key(labels)

            prev_vals = []
            if has_collision_key:
                prev_vals.append(int(labels.get("collision", 0)))
                labels["collision"] = lbl_val
            if has_col_key:
                prev_vals.append(int(labels.get("col", 0)))
                labels["col"] = lbl_val
            if not (has_collision_key or has_col_key):
                # If neither exists, create a conservative 'col' field
                prev_vals.append(None)
                labels["col"] = lbl_val
                rec["labels"] = labels

            # Track changes if any existing value differs
            for prev in prev_vals:
                if prev is None:
                    continue
                if prev != lbl_val:
                    changed += 1
                    if lbl_val == 1:
                        flips_to_one += 1
                    else:
                        flips_to_zero += 1
                    break

            fout.write(json.dumps(rec) + "\n")

    print(f"Processed: {total}")
    print(f"Changed:   {changed}")
    print(f"  → flips to 1 (no collision): {flips_to_one}")
    print(f"  → flips to 0 (collision):    {flips_to_zero}")
    print(f"Box-only cases encountered: {box_only_cases}")
    print(f"Wrote: {output_path}")


def main():
    in_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.waypoints.jsonl"
    out_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.fixed.jsonl"
    raw_root = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
    # process(in_path, out_path, raw_root)

    # --- Phase 2: sync experiments/new.jsonl labels from fixed waypoints ---
    # Build id -> col label map from fixed waypoints file (supports labels.col or labels.collision)
    id_to_col = {}
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id") or rec.get("ID")
            if not rid:
                continue
            labels = rec.get("labels", {})
            if "col" in labels:
                id_to_col[rid] = int(labels.get("col", 0))
            elif "collision" in labels:
                id_to_col[rid] = int(labels.get("collision", 0))

    exp_in = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/experiments/new.jsonl"
    exp_out = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/experiments/new.fixed.jsonl"

    os.makedirs(os.path.dirname(exp_out), exist_ok=True)
    total = 0
    updated = 0
    missing = 0
    with open(exp_in, "r") as fin, open(exp_out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            idx = rec.get("index")
            if idx is None:
                fout.write(json.dumps(rec) + "\n")
                continue
            sid = f"S{int(idx):05d}"
            if sid in id_to_col:
                labels = rec.get("labels", {})
                prev = labels.get("col")
                labels["col"] = int(id_to_col[sid])
                rec["labels"] = labels
                if prev is None or int(prev) != int(id_to_col[sid]):
                    updated += 1
            else:
                missing += 1
            fout.write(json.dumps(rec) + "\n")

    print(f"[experiments] processed={total} updated={updated} missing_ids={missing}")





if __name__ == "__main__":
    main()


