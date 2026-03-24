#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from placement_quality.path_simulation.model_training.model import (
    PickPlaceFeasibilityNet,
    PickPlaceDataset,
    wxyz_to_R,
    rot_to_6d,
    norm_pos_world,
    BOX_DIMS,
    corners_world,
)


# ---------- Fixed defaults (no argparse) ----------
DEFAULT_SIM_PATH = \
    "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.waypoints.jsonl"
DEFAULT_RESULTS_PATH = \
    "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/experiments/experiment_results_origin_box.jsonl"
DEFAULT_CHECKPOINT = \
    "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/abs_quat_corner_world/best.pt"
DEFAULT_SCORE_MODE = "combo"  # "ik_only" or "combo"

# Match training-time toggles
USE_CORNERS = True
USE_META    = True
USE_DELTA   = True

# Toggle (mirror train)
USE_TRANSPORT = True    # express grasp in transport frame

# Dataset paths for test evaluation (mirror train.py)
RAW_ROOT         = \
    "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = \
    "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
PAIRS_DIR        = \
    "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs"
PAIRS_TEST       = os.path.join(PAIRS_DIR, "pairs_test.jsonl")

def _compose_precomp_root():
    base = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/precomputed"
    parts = []
    if USE_DELTA:
        parts.append("delta")
    if USE_TRANSPORT:
        parts.append("transport")
    else:
        parts.append("original")
    if parts:
        return f"{base}_" + "_".join(parts)
    return base

PRECOMP_ROOT = _compose_precomp_root()


# ---------- Helpers ----------

def read_json_or_jsonl(path):
    items = []
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    return items


def build_waypoint_dict(case):
    wpd = {}
    for wp in case.get("waypoints", []):
        pose = wp.get("pose", {})
        wpd[wp.get("name")] = [pose.get("position"), pose.get("orientation_quat")]
    return wpd


def face_one_hot(face_str):
    mapping = {"+X":0, "-X":1, "+Y":2, "-Y":3}
    v = np.zeros(4, dtype=np.float32)
    if face_str in mapping:
        v[mapping[face_str]] = 1.0
    return v


def build_meta_from_case(case):
    g = case.get("grasp", {})
    u_frac = float(g.get("u_frac", 0.5))
    v_frac = float(g.get("v_frac", 0.5))
    onehot = face_one_hot(str(g.get("face", "")))
    meta = np.concatenate([[u_frac, v_frac], onehot], axis=0).astype(np.float32)
    return torch.from_numpy(meta).unsqueeze(0).float()


def build_corners_from_final_pose(pos_f, R_WOf):
    cf = corners_world(pos_f, R_WOf, BOX_DIMS).astype(np.float32)
    return torch.from_numpy(cf).unsqueeze(0).float()


def compute_features_from_case(case):
    # Object world poses at pick/place (wxyz quaternions)
    pos_i  = np.array(case["pick_object_world"]["position"], dtype=np.float32)
    quat_i = np.array(case["pick_object_world"]["orientation_quat"], dtype=np.float32)
    pos_f  = np.array(case["place_object_world"]["position"], dtype=np.float32)
    quat_f = np.array(case["place_object_world"]["orientation_quat"], dtype=np.float32)

    # Grasp contact pose in world (C1 waypoint)
    wpd = build_waypoint_dict(case)
    c1 = wpd.get("C1", None)
    if c1 is None:
        raise RuntimeError("Missing C1 waypoint in sim record")
    posCi = np.array(c1[0], dtype=np.float32)
    quatCi = np.array(c1[1], dtype=np.float32)

    R_WOi = wxyz_to_R(quat_i)
    R_WOf = wxyz_to_R(quat_f)
    R_WCi = wxyz_to_R(quatCi)

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
    if USE_DELTA:
        objW_place = np.concatenate([norm_pos_world(pos_f - pos_i), rot_to_6d(R_delta)], axis=0)
    else:
        objW_place = np.concatenate([norm_pos_world(pos_f), rot_to_6d(R_WOf)], axis=0)

    # Meta and corners
    meta = build_meta_from_case(case)
    corners_f = build_corners_from_final_pose(pos_f, R_WOf)

    return (
        torch.from_numpy(grasp_Oi).unsqueeze(0).float(),
        torch.from_numpy(objW_pick).unsqueeze(0).float(),
        torch.from_numpy(objW_place).unsqueeze(0).float(),
        meta,
        corners_f,
    )


def load_model(checkpoint_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer to read config from checkpoint to avoid shape mismatches
    ckpt_cfg = None
    ckpt_state_dict = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        raw = torch.load(checkpoint_path, map_location=device)
        if isinstance(raw, dict):
            ckpt_cfg = raw.get("cfg", None)
            if "state_dict" in raw:
                ckpt_state_dict = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], dict):
                ckpt_state_dict = raw["model"]
    use_meta_flag = bool(ckpt_cfg.get("USE_META", USE_META)) if isinstance(ckpt_cfg, dict) else USE_META
    use_corners_flag = bool(ckpt_cfg.get("USE_CORNERS", USE_CORNERS)) if isinstance(ckpt_cfg, dict) else USE_CORNERS

    model = PickPlaceFeasibilityNet(
        use_meta=use_meta_flag,
        use_corners=use_corners_flag,
        hidden=64,
        dropout=0.05,
    ).to(device)

    if ckpt_state_dict is not None:
        cleaned = {}
        for k, v in ckpt_state_dict.items():
            nk = k
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod."):]
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()
    return model, device


def score_from_logits(logit_ik, logit_col, mode):
    p_ik = torch.sigmoid(logit_ik)
    p_col_free = torch.sigmoid(logit_col)
    if mode == "ik_only":
        return p_ik.item(), p_ik.item(), p_col_free.item()
    # default: combined feasibility
    return (p_ik * (p_col_free)).item(), p_ik.item(), p_col_free.item()


# ---------- Simple analysis over in-memory results ----------

def summarize_predictions(results, cases):
    """Print accuracy using 0.5 thresholds.
    IK truth: feasible iff reason != "IK".
    Collision-free truth: no collision iff collision_counter == 0.
    """
    n = 0
    # IK (positive = feasible)
    ik_tp = ik_fp = ik_tn = ik_fn = 0
    # Collision-free (positive = collision-free)
    cf_tp = cf_fp = cf_tn = cf_fn = 0
    # Keep examples
    ik_fp_cases = []
    ik_fn_cases = []
    cf_fp_cases = []
    cf_fn_cases = []

    def _pose_fields(case):
        try:
            pi = case.get("pick_object_world", {})
            pf = case.get("place_object_world", {})
            ipos = [round(float(x), 3) for x in pi.get("position", [])]
            iquat = [round(float(x), 3) for x in pi.get("orientation_quat", [])]
            fpos = [round(float(x), 3) for x in pf.get("position", [])]
            fquat = [round(float(x), 3) for x in pf.get("orientation_quat", [])]
            return ipos, iquat, fpos, fquat
        except Exception:
            return [], [], [], []

    for r in results:
        p_ik = float(r.get("p_ik", 0.0))
        p_cf = float(r.get("p_collision", 0.0))  # this field stores P(collision-free)

        pred_ik_feasible = (p_ik >= 0.5)
        pred_collision_free = (p_cf >= 0.5)

        reason = r.get("reason", None)
        ik_true_feasible = (reason != "IK")

        collided = int(r.get("collision_counter", 0)) > 0
        cf_true = (not collided)

        # IK confusion
        if pred_ik_feasible and ik_true_feasible:
            ik_tp += 1
        elif pred_ik_feasible and not ik_true_feasible:
            ik_fp += 1
            idx = int(r.get("index", -1))
            ipos, iquat, fpos, fquat = _pose_fields(cases[idx]) if 0 <= idx < len(cases) else ([], [], [], [])
            ik_fp_cases.append({
                "index": r.get("index"),
                "p_ik": round(p_ik, 3),
                "reason": reason,
                "init_pos": ipos,
                "init_quat": iquat,
                "final_pos": fpos,
                "final_quat": fquat,
            })
        elif (not pred_ik_feasible) and (not ik_true_feasible):
            ik_tn += 1
        else:
            ik_fn += 1
            idx = int(r.get("index", -1))
            ipos, iquat, fpos, fquat = _pose_fields(cases[idx]) if 0 <= idx < len(cases) else ([], [], [], [])
            ik_fn_cases.append({
                "index": r.get("index"),
                "p_ik": round(p_ik, 3),
                "reason": reason,
                "init_pos": ipos,
                "init_quat": iquat,
                "final_pos": fpos,
                "final_quat": fquat,
            })

        # Collision-free confusion
        if pred_collision_free and cf_true:
            cf_tp += 1
        elif pred_collision_free and not cf_true:
            cf_fp += 1
            idx = int(r.get("index", -1))
            ipos, iquat, fpos, fquat = _pose_fields(cases[idx]) if 0 <= idx < len(cases) else ([], [], [], [])
            cf_fp_cases.append({
                "index": r.get("index"),
                "p_collision_free": round(p_cf, 3),
                "collision_counter": int(r.get("collision_counter", 0)),
                "init_pos": ipos,
                "init_quat": iquat,
                "final_pos": fpos,
                "final_quat": fquat,
            })
        elif (not pred_collision_free) and (not cf_true):
            cf_tn += 1
        else:
            cf_fn += 1
            idx = int(r.get("index", -1))
            ipos, iquat, fpos, fquat = _pose_fields(cases[idx]) if 0 <= idx < len(cases) else ([], [], [], [])
            cf_fn_cases.append({
                "index": r.get("index"),
                "p_collision_free": round(p_cf, 3),
                "collision_counter": int(r.get("collision_counter", 0)),
                "init_pos": ipos,
                "init_quat": iquat,
                "final_pos": fpos,
                "final_quat": fquat,
            })

        n += 1

    def _acc(tp, fp, tn, fn):
        total = tp + fp + tn + fn
        return (tp + tn) / max(1, total)

    ik_acc = _acc(ik_tp, ik_fp, ik_tn, ik_fn)
    cf_acc = _acc(cf_tp, cf_fp, cf_tn, cf_fn)
    both_correct = ik_tp if False else None  # placeholder to keep style consistent

    # Joint accuracy: both heads correct on the same sample
    joint_correct = 0
    for r in results:
        p_ik = float(r.get("p_ik", 0.0))
        p_cf = float(r.get("p_collision", 0.0))
        pred_ik = (p_ik >= 0.5)
        pred_cf = (p_cf >= 0.5)
        reason = r.get("reason", None)
        ik_true = (reason != "IK")
        collided = int(r.get("collision_counter", 0)) > 0
        cf_true = (not collided)
        if (pred_ik == ik_true) and (pred_cf == cf_true):
            joint_correct += 1
    joint_acc = joint_correct / max(1, n)

    print("[Analysis] samples=", n)
    print("IK: acc=", round(ik_acc, 4), "TP=", ik_tp, "FP=", ik_fp, "TN=", ik_tn, "FN=", ik_fn)
    print("Collision-free: acc=", round(cf_acc, 4), "TP=", cf_tp, "FP=", cf_fp, "TN=", cf_tn, "FN=", cf_fn)
    print("Joint (both correct): acc=", round(joint_acc, 4))

    # Print a few examples for FP/FN
    def _preview(lst, k=20):
        return lst[:min(k, len(lst))]
    if ik_fp_cases:
        print("IK FP cases (pred feasible, true not feasible):")
        for c in _preview(ik_fp_cases):
            print(f"  index={c.get('index')}  p_ik={c.get('p_ik')}  reason={c.get('reason')}  "
                  f"init_pos={c.get('init_pos')} init_quat={c.get('init_quat')}  final_pos={c.get('final_pos')} final_quat={c.get('final_quat')}")
        print("-" * 80)
    if ik_fn_cases:
        print("IK FN cases (pred not feasible, true feasible):")
        for c in _preview(ik_fn_cases):
            print(f"  index={c.get('index')}  p_ik={c.get('p_ik')}  reason={c.get('reason')}  "
                  f"init_pos={c.get('init_pos')} init_quat={c.get('init_quat')}  final_pos={c.get('final_pos')} final_quat={c.get('final_quat')}")
        print("-" * 80)
    if cf_fp_cases:
        print("Collision-free FP cases (pred no collision, true collided):")
        for c in _preview(cf_fp_cases):
            print(f"  index={c.get('index')}  p_cf={c.get('p_collision_free')}  collision_counter={c.get('collision_counter')}  "
                  f"init_pos={c.get('init_pos')} init_quat={c.get('init_quat')}  final_pos={c.get('final_pos')} final_quat={c.get('final_quat')}")
        print("-" * 80)
    if cf_fn_cases:
        print("Collision-free FN cases (pred collision, true no collision):")
        for c in _preview(cf_fn_cases):
            print(f"  index={c.get('index')}  p_cf={c.get('p_collision_free')}  collision_counter={c.get('collision_counter')}  "
                  f"init_pos={c.get('init_pos')} init_quat={c.get('init_quat')}  final_pos={c.get('final_pos')} final_quat={c.get('final_quat')}")
        print("-" * 80)


# ---------- Test split evaluation (like train.py end) ----------

def evaluate_test_split(checkpoint_path):
    print("\n[Eval] Running test-split evaluation…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test dataset (prefer precomputed if present)
    precomp_dir = os.path.join(PRECOMP_ROOT, "test")
    use_precomp = os.path.isdir(precomp_dir)
    ds_test = PickPlaceDataset(
        PAIRS_TEST,
        RAW_ROOT,
        GRASPS_META_PATH,
        use_corners=USE_CORNERS,
        use_meta=USE_META,
        use_delta=USE_DELTA,
        precomp_dir=precomp_dir if use_precomp else None,
    )

    model, device = load_model(checkpoint_path, device)

    model.eval()

    dl = DataLoader(ds_test, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)

    n = 0
    corr_ik = corr_col = 0
    with torch.no_grad():
        for batch in dl:
            grasp_Oi   = batch["grasp_Oi"].float().to(device)
            objW_pick  = batch["objW_pick"].float().to(device)
            objW_place = batch["objW_place"].float().to(device)
            meta       = batch.get("meta", None)
            corners_f  = batch.get("corners_f", None)
            B = grasp_Oi.shape[0]
            if meta is None:
                meta = torch.zeros(B, 6, dtype=torch.float32, device=device)
            else:
                meta = meta.float().to(device)
            if corners_f is None:
                corners_f = torch.zeros(B, 24, dtype=torch.float32, device=device)
            else:
                corners_f = corners_f.float().to(device)

            li, lc = model(
                grasp_Oi,
                objW_pick,
                objW_place,
                meta=meta,
                corners_f=corners_f,
            )

            yik  = batch["y_ik"].to(device)
            ycol = batch["y_col"].to(device)

            pred_ik  = (torch.sigmoid(li)  >= 0.5).float()
            pred_col = (torch.sigmoid(lc)  >= 0.5).float()

            corr_ik  += (pred_ik  == yik).sum().item()
            corr_col += (pred_col == ycol).sum().item()
            n        += yik.shape[0]

    acc_ik  = corr_ik  / max(1, n)
    acc_col = corr_col / max(1, n)
    print(f"[TEST] acc_ik={acc_ik:.4f}  acc_col={acc_col:.4f}")


# ---------- Analysis from results JSONL ----------

def analyze_results(results_path):
    data = read_json_or_jsonl(results_path)
    # IK ground-truth: IK-feasible iff reason != "IK"
    # Collision ground-truth: collided iff collision_counter > 0
    ik_tp = ik_fp = ik_tn = ik_fn = 0
    col_tp = col_fp = col_tn = col_fn = 0

    # Labels vs actual
    ik_lab_matches = ik_lab_mismatches = 0
    col_lab_matches = col_lab_mismatches = 0

    sample_mismatch_ik = []
    sample_mismatch_col = []

    for r in data:
        p_ik = float(r.get("p_ik", 0.0))
        p_col = float(r.get("p_collision", 0.0))
        ik_pred = (p_ik >= 0.5)
        col_pred = (p_col >= 0.5)

        reason = r.get("reason", None)
        is_ik_feasible = (reason != "IK")
        ik_true = is_ik_feasible  # True means feasible

        collided = int(r.get("collision_counter", 0)) > 0
        col_true = collided  # True means collision occurred

        # IK metrics: positive=feasible
        if ik_pred and ik_true:
            ik_tp += 1
        elif ik_pred and not ik_true:
            ik_fp += 1
        elif not ik_pred and not ik_true:
            ik_tn += 1
        else:
            ik_fn += 1

        # Collision metrics: positive=collision
        if col_pred and col_true:
            col_tp += 1
        elif col_pred and not col_true:
            col_fp += 1
        elif not col_pred and not col_true:
            col_tn += 1
        else:
            col_fn += 1

        # Labels extracted during scoring from SIM deck
        lab = r.get("labels", None)
        if isinstance(lab, dict):
            lab_ik = int(lab.get("ik", 0))  # 1 means feasible per dataset label
            lab_col = int(lab.get("col", 0))  # 1 means collision per dataset label
            # Compare labels vs actual experiment outcomes
            if (lab_ik == 1) == ik_true:
                ik_lab_matches += 1
            else:
                ik_lab_mismatches += 1
                if len(sample_mismatch_ik) < 5:
                    sample_mismatch_ik.append({"index": r.get("index"), "label_ik": lab_ik, "actual_feasible": ik_true, "reason": reason})
            if (lab_col == 0) == col_true:
                col_lab_matches += 1
            else:
                col_lab_mismatches += 1
                if len(sample_mismatch_col) < 5:
                    sample_mismatch_col.append({"index": r.get("index"), "label_col": lab_col, "actual_collision": col_true, "collision_counter": r.get("collision_counter")})

    def summarize(tp, fp, tn, fn, name):
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(1, total)
        print(f"[{name}] total={total} acc={acc:.4f} TP={tp} FP={fp} TN={tn} FN={fn}")

    summarize(ik_tp, ik_fp, ik_tn, ik_fn, "IK(feasible)")
    summarize(col_tp, col_fp, col_tn, col_fn, "Collision")

    total = max(1, ik_lab_matches + ik_lab_mismatches)
    print(f"[Labels vs Actual] IK matches={ik_lab_matches} mismatches={ik_lab_mismatches} (match_rate={(ik_lab_matches/total):.4f})")
    total = max(1, col_lab_matches + col_lab_mismatches)
    print(f"[Labels vs Actual] Collision matches={col_lab_matches} mismatches={col_lab_mismatches} (match_rate={(col_lab_matches/total):.4f})")
    if sample_mismatch_ik:
        print("Sample IK label/actual mismatches:")
        for s in sample_mismatch_ik:
            print("  ", s)
    if sample_mismatch_col:
        print("Sample Collision label/actual mismatches:")
        for s in sample_mismatch_col:
            print("  ", s)


# ---------- Main: fill results + optional test eval ----------

def main():
    sim_path     = DEFAULT_SIM_PATH
    results_path = DEFAULT_RESULTS_PATH
    checkpoint   = DEFAULT_CHECKPOINT
    score_mode   = DEFAULT_SCORE_MODE

    # Load SIM cases and results
    cases = read_json_or_jsonl(sim_path)
    with open(results_path, "r") as f:
        results_lines = [line.rstrip("\n") for line in f if line.strip()]
    results = [json.loads(line) for line in results_lines]

    # Load model and align rotation representation from checkpoint
    model, device = load_model(checkpoint)
    global PRECOMP_ROOT
    PRECOMP_ROOT = _compose_precomp_root()

    # Compute and overwrite scores unconditionally
    updated_lines = []
    with torch.no_grad():
        for rec in results:
            idx = int(rec["index"])  # 0-based index into SIM cases
            case = cases[idx]
            grasp_Oi, objW_pick, objW_place, meta, corners_f = compute_features_from_case(case)
            grasp_Oi = grasp_Oi.to(device)
            objW_pick = objW_pick.to(device)
            objW_place = objW_place.to(device)
            meta = meta.to(device)
            corners_f = corners_f.to(device)
            logit_ik, logit_col = model(grasp_Oi, objW_pick, objW_place, meta=meta, corners_f=corners_f)
            score, p_ik, p_col = score_from_logits(logit_ik, logit_col, score_mode)
            rec["prediction_score"] = float(score)
            rec["p_ik"] = float(p_ik)
            rec["p_collision"] = float(p_col)
            # attach ground-truth labels if present in case
            labels = case.get("labels", None)
            if isinstance(labels, dict):
                rec["labels"] = {"ik": int(labels.get("ik", 0)), "col": int(labels.get("col", 0))}
            updated_lines.append(json.dumps(rec))

    # Print simple analysis using 0.5 thresholds
    summarize_predictions(results, cases)

    # Overwrite results file in-place
    # with open(results_path, "w") as f:
    #     for line in updated_lines:
    #         f.write(line + "\n")

        
    # evaluate_test_split(checkpoint)

    # Print analysis using 0.5 thresholds and labels-vs-actual
    analyze_results(results_path)


if __name__ == "__main__":
    main()