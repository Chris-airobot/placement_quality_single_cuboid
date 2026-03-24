import os, json
from placement_quality.path_simulation.model_training.post_processing import RAW_ROOT


def _load_endpoint_row(p, g, o):
    folder = os.path.join(RAW_ROOT, f"p{int(p)}")
    fp = os.path.join(folder, f"data_{int(g)}.json")
    with open(fp, "r") as f:
        data = json.load(f)
    row = data[str(int(o))]
    return row, fp


def _col_label_from_row(row):
    any_flag = bool(row.get("endpoint_collision_at_C", {}).get("any", False))
    col_free = (not any_flag)
    return 1 if col_free else 0, any_flag, col_free


def main():
    pick = {"p": 3, "g": 776, "o": 112}
    place = {"p": 5, "g": 776, "o": 40}

    pick_row, pick_path = _load_endpoint_row(pick["p"], pick["g"], pick["o"])
    place_row, place_path = _load_endpoint_row(place["p"], place["g"], place["o"])

    print("Pick source:")
    print(f"  file: {pick_path}")
    print(f"  endpoint_collision_at_C: {pick_row.get('endpoint_collision_at_C')}")

    print("Place target:")
    print(f"  file: {place_path}")
    print(f"  endpoint_collision_at_C: {place_row.get('endpoint_collision_at_C')}")

    col_lbl, any_flag, col_free = _col_label_from_row(place_row)
    print("Derived label (from place endpoint):")
    print(f"  any_collision_at_C = {any_flag}")
    print(f"  collision_free_at_C = {col_free}")
    print(f"  collision label = {col_lbl} (1 means collision-free at contact)")


if __name__ == "__main__":
    main()


