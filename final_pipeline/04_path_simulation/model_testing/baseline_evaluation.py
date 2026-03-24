file_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/experiments/comparison_results.jsonl"

import json

max_cases = 100  # set to an integer (e.g., 500) to evaluate only the first K cases

def _is_success(record):
    # Success only if no collisions and reason is a dict with 'position_error'
    if record.get("collision_counter", 0) != 0:
        return False
    reason = record.get("reason")
    if not isinstance(reason, dict):
        return False
    return "position_error" in reason


def _append_attempt(case_map, record):
    labels = record.get("labels", {})
    case_index = labels.get("case_index")
    policy = labels.get("policy")
    if case_index is None or policy not in ("B", "M"):
        return
    by_policy = case_map.setdefault(case_index, {"B": [], "M": []})
    # Capture a stable case identifier
    case_id = record.get("case_id") or labels.get("trial_id") or str(case_index)
    if "_case_id" not in by_policy:
        by_policy["_case_id"] = case_id

    exec_num = record.get("execution_number")
    if exec_num is None:
        ai = labels.get("attempt_index")
        if ai is not None:
            exec_num = int(ai) + 1
        else:
            exec_num = len(by_policy[policy]) + 1

    by_policy[policy].append({
        "execution_number": int(exec_num),
        "success": _is_success(record),
    })


def _first_success_index(attempts):
    if not attempts:
        return None
    # Use read/insertion order so earlier failures count toward the attempt index
    for idx, a in enumerate(attempts):
        if a["success"]:
            return idx + 1
    return None

def main():
    case_map = {}
    selected_cases = []
    selected_set = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = rec.get("labels", {})
            ci = labels.get("case_index")
            pol = labels.get("policy")
            if ci is None or pol not in ("B", "M"):
                continue
            if max_cases is not None:
                if ci not in selected_set:
                    if len(selected_set) >= max_cases:
                        continue
                    selected_set.add(ci)
                    selected_cases.append(ci)
            _append_attempt(case_map, rec)

    case_indices = sorted(case_map.keys())

    exact_counts = {"B": {1: 0, 2: 0, 3: 0}, "M": {1: 0, 2: 0, 3: 0}}
    baseline_win_case_ids = []
    tie_case_ids = []

    for ci in case_indices:
        attempts_B = case_map[ci]["B"]
        attempts_M = case_map[ci]["M"]
        case_id = case_map[ci].get("_case_id", str(ci))

        k_B = _first_success_index(attempts_B)
        k_M = _first_success_index(attempts_M)

        if k_B in (1, 2, 3):
            exact_counts["B"][k_B] += 1
        if k_M in (1, 2, 3):
            exact_counts["M"][k_M] += 1
        elif k_M is None and len(attempts_M) >= 3:
            # Model attempted three times and all failed → count as a model k1 win (predicting failure)
            exact_counts["M"][1] += 1

        if k_B is None and k_M is None:
            tie_case_ids.append(case_id)
        elif k_B is None:
            pass  # model wins; no print requested here
        elif k_M is None:
            baseline_win_case_ids.append(case_id)
        else:
            if k_B < k_M:
                baseline_win_case_ids.append(case_id)
            elif k_B == k_M:
                tie_case_ids.append(case_id)

    n_cases = len(case_indices)
    print("Cases:", n_cases)
    print("Exact@k counts — B:", f"k1={exact_counts['B'][1]}", f"k2={exact_counts['B'][2]}", f"k3={exact_counts['B'][3]}")
    print("Exact@k counts — M:", f"k1={exact_counts['M'][1]}", f"k2={exact_counts['M'][2]}", f"k3={exact_counts['M'][3]}")
    if baseline_win_case_ids:
        print("Baseline win case_ids:", ", ".join(map(str, baseline_win_case_ids)))
    if tie_case_ids:
        print("Tie case_ids:", ", ".join(map(str, tie_case_ids)))


if __name__ == "__main__":
    main()
