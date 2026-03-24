import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


# World frame constants
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
WORLD_DOWN = -WORLD_UP


# Local surface frames for each cuboid face in the OBJECT frame
_FACE_BASIS = {
    '+X': (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    '-X': (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), -np.array([1.0, 0.0, 0.0])),
    '+Y': (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    '-Y': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), -np.array([0.0, 1.0, 0.0])),
    '+Z': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    '-Z': (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), -np.array([0.0, 0.0, 1.0])),
}


def face_surface_frame(face: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_f, v_f, n_f = _FACE_BASIS[face]
    return u_f.copy(), v_f.copy(), n_f.copy()


def quat_wxyz_to_xyzw(q: List[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(q: np.ndarray) -> List[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def rotation_matrix_from_quat_wxyz(q_wxyz: List[float]) -> np.ndarray:
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    return R.from_quat(q_xyzw).as_matrix()


def quat_wxyz_from_rotation_matrix(R_wo: np.ndarray) -> List[float]:
    q_xyzw = R.from_matrix(R_wo).as_quat()
    return quat_xyzw_to_wxyz(q_xyzw)


def load_pedestals(pedestal_json_path: str) -> List[Dict]:
    with open(pedestal_json_path, "r") as f:
        data = json.load(f)
    # Expect list of {id, position[x,y,z], orientation[x,y,z,w]}
    return data


def compute_pedestal_bins(pedestals: List[Dict], robot_center_xy: Tuple[float, float]) -> Dict[str, List[int]]:
    # Row-based binning (top->bottom rows). Central row(s) = close, next rows = mid, outer = far.
    # Group pedestals by unique Y values (rows), then classify rows.
    # This matches the 2-column x 5-row (or 6-row) layout better than distance tertiles.

    # Build mapping from row_y to indices in that row
    row_to_indices: Dict[float, List[int]] = {}
    for idx, p in enumerate(pedestals):
        y = float(p["position"][1])
        row_to_indices.setdefault(y, []).append(idx)

    # Sort rows by Y ascending (top to bottom consistent with earlier grid mapping)
    sorted_rows: List[Tuple[float, List[int]]] = sorted(row_to_indices.items(), key=lambda kv: kv[0])
    num_rows = len(sorted_rows)

    # Determine central row(s)
    if num_rows % 2 == 1:
        # Odd rows: single central row is 'close'
        center_start = num_rows // 2
        close_rows = {center_start}
    else:
        # Even rows: two central rows are 'close'
        center_start = num_rows // 2 - 1
        close_rows = {center_start, center_start + 1}

    # Mid rows: the immediate neighbors around close rows (up to two rows total if available)
    mid_rows: set = set()
    k = 1
    while len(mid_rows) < 2 and (len(close_rows) + len(mid_rows)) < num_rows:
        left = center_start - k
        right = (center_start + (0 if num_rows % 2 == 1 else 1)) + k
        if left >= 0 and left not in close_rows:
            mid_rows.add(left)
        if right < num_rows and right not in close_rows and len(mid_rows) < 2:
            mid_rows.add(right)
        k += 1

    # Far rows: remaining
    all_rows = set(range(num_rows))
    far_rows = all_rows - close_rows - mid_rows

    bins = {"close": [], "mid": [], "far": []}
    for r_idx in sorted(close_rows):
        bins["close"].extend(sorted_rows[r_idx][1])
    for r_idx in sorted(mid_rows):
        bins["mid"].extend(sorted_rows[r_idx][1])
    for r_idx in sorted(far_rows):
        bins["far"].extend(sorted_rows[r_idx][1])

    return bins


def choose_index_from_bin(bins: Dict[str, List[int]], bin_name: str, rng: np.random.Generator) -> int:
    candidates = bins.get(bin_name, [])
    if len(candidates) == 0:
        # Fallback to any available
        all_idxs = bins.get("close", []) + bins.get("mid", []) + bins.get("far", [])
        return int(rng.choice(all_idxs))
    return int(rng.choice(candidates))


def nearest_pedestal_index(pedestals: List[Dict], position_xyz: Tuple[float, float, float]) -> int:
    px, py = float(position_xyz[0]), float(position_xyz[1])
    best_i = 0
    best_d = float("inf")
    for idx, p in enumerate(pedestals):
        x, y = float(p["position"][0]), float(p["position"][1])
        d = math.hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            best_i = idx
    return best_i


def compute_pedestal_bins_relative(pedestals: List[Dict], source_idx: int) -> Dict[str, List[int]]:
    # Sort other pedestals by XY distance from source; split into equal-count bins (close/mid/far)
    sx, sy = float(pedestals[source_idx]["position"][0]), float(pedestals[source_idx]["position"][1])
    items: List[Tuple[int, float]] = []
    for idx, p in enumerate(pedestals):
        if idx == int(source_idx):
            continue
        x, y = float(p["position"][0]), float(p["position"][1])
        d = math.hypot(x - sx, y - sy)
        items.append((idx, d))
    items.sort(key=lambda t: t[1])

    n = len(items)
    # Aim for balanced 3-3-? split; for 9 items -> 3/3/3; for 10+ exclude source already
    k = n // 3
    r = n - 2 * k
    close_cut = k
    mid_cut = 2 * k
    # Put remainder into far to ensure close is tightest
    close_idxs = [idx for (idx, _) in items[:close_cut]]
    mid_idxs = [idx for (idx, _) in items[close_cut:mid_cut]]
    far_idxs = [idx for (idx, _) in items[mid_cut:]]

    return {"close": close_idxs, "mid": mid_idxs, "far": far_idxs}


def sorted_grid_indices(pedestals: List[Dict]) -> List[int]:
    # Sort by Y (top to bottom: low->high), then X (left->right: low->high)
    idxs = list(range(len(pedestals)))
    idxs.sort(key=lambda i: (float(pedestals[i]["position"][1]), float(pedestals[i]["position"][0])))
    return idxs


def grid_index_maps(pedestals: List[Dict]) -> Tuple[List[int], Dict[int, int]]:
    order = sorted_grid_indices(pedestals)
    inv = {int(ped_idx): int(grid_idx) for grid_idx, ped_idx in enumerate(order)}
    return order, inv


def verify_bins_from_grid_index(
    pedestals: List[Dict],
    grid_index: int,
    robot_center_xy: Tuple[float, float] = (0.0, 0.0),
    rng_seed: Optional[int] = None,
) -> Dict:
    rng = np.random.default_rng(rng_seed)
    order, inv = grid_index_maps(pedestals)
    # Map provided grid index to pedestal record
    ped_idx = int(order[grid_index])
    # Bins relative to the selected pedestal
    bins = compute_pedestal_bins_relative(pedestals, source_idx=ped_idx)

    # Sample one from each bin
    pick_close = choose_index_from_bin(bins, "close", rng)
    pick_mid = choose_index_from_bin(bins, "mid", rng)
    pick_far = choose_index_from_bin(bins, "far", rng)

    # Convert to grid numbering
    out = {
        "input": {
            "grid_index": int(grid_index),
            "pedestal_id": pedestals[ped_idx].get("id", str(ped_idx)),
        },
        "ordering_grid_to_id": [pedestals[i].get("id", str(i)) for i in order],
        "choices": {
            "close": {
                "grid_index": int(inv[int(pick_close)]),
                "pedestal_id": pedestals[int(pick_close)].get("id", str(pick_close)),
            },
            "mid": {
                "grid_index": int(inv[int(pick_mid)]),
                "pedestal_id": pedestals[int(pick_mid)].get("id", str(pick_mid)),
            },
            "far": {
                "grid_index": int(inv[int(pick_far)]),
                "pedestal_id": pedestals[int(pick_far)].get("id", str(pick_far)),
            },
        },
    }
    return out


def detect_down_face_label(R_wo_init: np.ndarray) -> str:
    # World down expressed in object frame
    n_down_obj = R_wo_init.T @ WORLD_DOWN
    ax = int(np.argmax(np.abs(n_down_obj)))
    sign = 1.0 if n_down_obj[ax] >= 0.0 else -1.0
    if ax == 0:
        return "+X" if sign > 0.0 else "-X"
    if ax == 1:
        return "+Y" if sign > 0.0 else "-Y"
    return "+Z" if sign > 0.0 else "-Z"


def long_axis_label(dims_xyz: Tuple[float, float, float]) -> str:
    dx, dy, _ = float(dims_xyz[0]), float(dims_xyz[1]), float(dims_xyz[2])
    return "X" if dx >= dy else "Y"


def build_rotation_mapping_face_to_down(face_label: str) -> np.ndarray:
    # Build world basis: choose in-plane X axis in world, and -Z for normal (down)
    u_t = np.array([1.0, 0.0, 0.0], dtype=float)
    n_t = WORLD_DOWN.copy()
    v_t = np.cross(n_t, u_t)
    v_t = v_t / np.linalg.norm(v_t)
    B_world = np.column_stack([u_t, v_t, n_t])

    u_f, v_f, n_f = face_surface_frame(face_label)
    B_obj = np.column_stack([u_f, v_f, n_f])
    # Map object surface frame to world target frame
    R_wo = B_world @ B_obj.T
    return R_wo


def sample_target_face_label(
    orientation_type: str,
    R_wo_init: np.ndarray,
    dims_xyz: Tuple[float, float, float],
    rng: np.random.Generator,
) -> str:
    down_face = detect_down_face_label(R_wo_init)

    def axis_pair_from_vector(v: np.ndarray) -> Tuple[str, str]:
        v = np.asarray(v, dtype=float)
        ax = int(np.argmax(np.abs(v)))
        if ax == 0:
            return "+X", "-X"
        if ax == 1:
            return "+Y", "-Y"
        return "+Z", "-Z"

    if orientation_type == "zup":
        # Randomly choose +Z or -Z as the down face (user preference)
        return str(rng.choice(["+Z", "-Z"]))

    if orientation_type == "adjacent":
        # Randomly choose left/right relative to the current down face,
        # defined by the down face's local u-axis in object frame.
        u_f, _, _ = face_surface_frame(down_face)
        f_pos, f_neg = axis_pair_from_vector(u_f)
        return str(rng.choice([f_pos, f_neg]))

    if orientation_type == "opposite":
        # Fixed: choose the + direction of the object's long axis
        axis = long_axis_label(dims_xyz)
        return "+X" if axis == "X" else "+Y"

    # Default fallback (should not happen if inputs are valid)
    return "-Z"


def object_top_height_above_pedestal(R_wo: np.ndarray, dims_xyz: Tuple[float, float, float]) -> float:
    half_extents = 0.5 * np.asarray(dims_xyz, dtype=float)
    # Project half-extents along world Z to get vertical half-height
    return float(np.sum(np.abs(R_wo[2, :]) * half_extents))


def sample_final_pose(
    initial_position_xyz: Tuple[float, float, float],
    initial_quaternion_wxyz: Tuple[float, float, float, float],
    dims_xyz: Tuple[float, float, float],
    orientation_type: str,  # {'adjacent','opposite','zup'}
    distance_bin: str,      # {'close','mid','far'}
    pedestal_json_path: str,
    robot_center_xy: Tuple[float, float] = (0.0, 0.0),
    rng_seed: Optional[int] = None,
) -> Dict:
    rng = np.random.default_rng(rng_seed)

    pedestals = load_pedestals(pedestal_json_path)
    # Determine current pedestal by nearest to initial position (XY)
    source_idx = nearest_pedestal_index(pedestals, initial_position_xyz)
    # Build bins relative to current pedestal
    bins = compute_pedestal_bins_relative(pedestals, source_idx=source_idx)
    ped_idx = choose_index_from_bin(bins, distance_bin, rng)
    pedestal = pedestals[ped_idx]

    # Initial rotation
    R_wo_init = rotation_matrix_from_quat_wxyz(list(initial_quaternion_wxyz))

    # Choose target object face to be DOWN on the target pedestal
    target_face = sample_target_face_label(orientation_type, R_wo_init, dims_xyz, rng)
    R_wo_final = build_rotation_mapping_face_to_down(target_face)

    # Position on pedestal top with correct height
    ped_pos = np.asarray(pedestal["position"], dtype=float)
    ped_top_z = float(ped_pos[2])  # given positions are pedestal centers; assume z is top? adjust if needed
    # If given z is center height, uncomment the following line and set pedestal height accordingly
    # ped_top_z = float(ped_pos[2]) + 0.5 * PEDESTAL_HEIGHT
    h_z = object_top_height_above_pedestal(R_wo_final, dims_xyz)
    t_final = np.array([ped_pos[0], ped_pos[1], ped_top_z + h_z], dtype=float)

    q_final_wxyz = quat_wxyz_from_rotation_matrix(R_wo_final)

    return {
        "pedestal_id": pedestal.get("id", str(ped_idx)),
        "orientation_type": orientation_type,
        "distance_bin": distance_bin,
        "final_position": [float(t_final[0]), float(t_final[1]), float(t_final[2])],
        "final_orientation_wxyz": [float(q_final_wxyz[0]), float(q_final_wxyz[1]), float(q_final_wxyz[2]), float(q_final_wxyz[3])],
        "target_face_down": target_face,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sample final object poses for tasks")
    parser.add_argument("--pedestals", type=str, default="/home/chris/Chris/placement_ws/src/pedestal_poses.json")
    parser.add_argument("--robot-center", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--dims", type=float, nargs=3, default=(0.143, 0.0915, 0.051), metavar=("DX", "DY", "DZ"))
    parser.add_argument("--initial-pos", type=float, nargs=3, default=(0.2, -0.3, 0.1), metavar=("X", "Y", "Z"))
    parser.add_argument("--initial-quat-wxyz", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--orientation-type", type=str, choices=["adjacent", "opposite", "zup"], default="adjacent")
    parser.add_argument("--distance-bin", type=str, choices=["close", "mid", "far"], default="close")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="/home/chris/Chris/placement_ws/src/final_pose_sample.json")
    parser.add_argument("--verify-from-index", type=int, default=None,
                        help="If set (0-9), verify pedestal bins by sampling one index from each bin and printing indices in grid numbering.")

    args = parser.parse_args()

    # Verification path: sample pedestal numbers per bin in grid numbering (0..9)
    if args.verify_from_index is not None:
        pedestals = load_pedestals(args.pedestals)
        res = verify_bins_from_grid_index(
            pedestals=pedestals,
            grid_index=int(args.verify_from_index),
            robot_center_xy=(args.robot_center[0], args.robot_center[1]),
            rng_seed=args.seed,
        )
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return

    result = sample_final_pose(
        initial_position_xyz=(args.initial_pos[0], args.initial_pos[1], args.initial_pos[2]),
        initial_quaternion_wxyz=(args.initial_quat_wxyz[0], args.initial_quat_wxyz[1], args.initial_quat_wxyz[2], args.initial_quat_wxyz[3]),
        dims_xyz=(args.dims[0], args.dims[1], args.dims[2]),
        orientation_type=args.orientation_type,
        distance_bin=args.distance_bin,
        pedestal_json_path=args.pedestals,
        robot_center_xy=(args.robot_center[0], args.robot_center[1]),
        rng_seed=args.seed,
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


