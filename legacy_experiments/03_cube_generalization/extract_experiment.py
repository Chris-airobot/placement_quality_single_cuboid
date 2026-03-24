import os
import json
import numpy as np
from dataset import quat_wxyz_to_R_batched, r6_to_R_batched
from scipy.spatial.transform import Rotation as R
# ========= SUBSET EXPORT (SIM 10k; REAL 1k subset of SIM) =========
def _face_id_from_R_batched(Rm: np.ndarray) -> np.ndarray:
    # Map axis with largest |z| to face id: +X=1, +Y=2, -X=3, -Y=4, +Z=5, -Z=6
    zcols = np.abs(Rm[:, 2, :])                 # (B,3)
    col = np.argmax(zcols, axis=1)              # (B,)  0:X,1:Y,2:Z
    B = Rm.shape[0]
    sign = np.sign(Rm[np.arange(B), 2, col])    # (B,) sign of z at chosen column

    fid = np.empty(B, dtype=np.int16)
    m0 = (col == 0)
    m1 = (col == 1)
    m2 = (col == 2)
    fid[m0] = np.where(sign[m0] > 0, 1, 3)
    fid[m1] = np.where(sign[m1] > 0, 2, 4)
    fid[m2] = np.where(sign[m2] > 0, 5, 6)
    return fid

def _quat_angle_deg_abs_batched(qi: np.ndarray, qf: np.ndarray) -> np.ndarray:
    # qi,qf: (B,4) wxyz, not necessarily unit
    qi = qi / (np.linalg.norm(qi, axis=1, keepdims=True) + 1e-12)
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
    dots = np.abs(np.sum(qi*qf, axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))

def _classify_buckets(init7: np.ndarray, final7: np.ndarray, small_thresh_deg: float = 45.0) -> np.ndarray:
    Rf = quat_wxyz_to_R_batched(final7[:,3:7].copy())
    Ro = quat_wxyz_to_R_batched(init7[:,3:7].copy())
    fi = _face_id_from_R_batched(Ro)
    ff = _face_id_from_R_batched(Rf)
    same = (fi == ff)
    opp_map = {1:3, 3:1, 2:4, 4:2, 5:6, 6:5}
    opp = np.array([opp_map[int(x)] for x in fi], dtype=np.int16)
    opposite = (ff == opp)
    adjacent = (~same) & (~opposite)
    # small/medium split only for SAME
    ang = _quat_angle_deg_abs_batched(init7[:,3:7], final7[:,3:7])
    small = same & (ang <= small_thresh_deg)
    medium= same & (~small)
    out = np.empty(fi.shape[0], dtype=object)
    out[small] = "SMALL"; out[medium] = "MEDIUM"; out[adjacent] = "ADJACENT"; out[opposite] = "OPPOSITE"
    return out

def _wxyz_from_R_batch(Rm: np.ndarray) -> np.ndarray:
    B = Rm.shape[0]
    q_xyzw = R.from_matrix(Rm).as_quat()            # (B,4) xyzw
    q_wxyz = np.stack([q_xyzw[:,3], q_xyzw[:,0], q_xyzw[:,1], q_xyzw[:,2]], axis=1).astype(np.float32)
    return q_wxyz

def export_test_subsets(memmap_dir: str,
                        out_sim_jsonl: str,
                        out_real_jsonl: str,
                        total_sim: int = 10_000,
                        total_real: int = 1_000,
                        boxes_sim: int = 10,
                        boxes_real: int = 5,
                        bucket_targets = {"SMALL":0.2, "MEDIUM":0.3, "ADJACENT":0.3, "OPPOSITE":0.2},
                        eps: float = 1e-6,
                        seed: int = 42):
    rng = np.random.default_rng(seed)

    meta = json.load(open(os.path.join(memmap_dir, "meta.json"), "r"))
    N = int(meta["N"])
    mm_dims   = np.memmap(meta["dims_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_final7 = np.memmap(meta["final7_file"], dtype=np.float32, mode="r", shape=(N,7))
    mm_init7  = np.memmap(meta["init7_file"],  dtype=np.float32, mode="r", shape=(N,7))
    mm_tloc3  = np.memmap(meta["tloc_file"],   dtype=np.float32, mode="r", shape=(N,3))
    mm_rloc6  = np.memmap(meta["rloc6_file"],  dtype=np.float32, mode="r", shape=(N,6))
    mm_y      = np.memmap(meta["label_file"],  dtype=np.float32, mode="r", shape=(N,1))

    # choose boxes: spread over max(dim)
    dims_all = mm_dims.copy()
    size = dims_all.max(axis=1)
    order = np.argsort(size)
    picks = np.linspace(0, N-1, boxes_sim, dtype=int)
    box_idx = order[picks]
    box_keys = []
    for i in box_idx:
        d = mm_dims[i]
        key = (round(float(d[0]),6), round(float(d[1]),6), round(float(d[2]),6))
        box_keys.append(key)
    # unique
    seen=set(); box_keys=[k for k in box_keys if not (k in seen or seen.add(k))]
    if len(box_keys) < boxes_sim:
        # backfill random unique
        uni = np.unique(np.round(dims_all,6), axis=0)
        rng.shuffle(uni)
        for r in uni:
            k=(float(r[0]),float(r[1]),float(r[2]))
            if k not in seen:
                box_keys.append(k); seen.add(k)
            if len(box_keys)>=boxes_sim: break

    # allocate per box
    per_box_sim = total_sim // boxes_sim
    per_box_real= total_real // boxes_real
    buckets = ("SMALL","MEDIUM","ADJACENT","OPPOSITE")

    sim_rows = []
    # precompute buckets for all rows? too big; compute per box mask
    for bi, key in enumerate(box_keys):
        m = (np.abs(mm_dims[:,0]-key[0])<=eps) & (np.abs(mm_dims[:,1]-key[1])<=eps) & (np.abs(mm_dims[:,2]-key[2])<=eps)
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        init7 = mm_init7[idx]
        final7= mm_final7[idx]
        buck  = _classify_buckets(init7, final7)
        # target counts
        targets = {b: int(round(per_box_sim * bucket_targets[b])) for b in buckets}
        # sample per bucket (initial)
        chosen_idx_list = []
        chosen_counts = {b:0 for b in buckets}
        pool_by_bucket = {}
        for b in buckets:
            sel = idx[buck == b]
            pool_by_bucket[b] = sel
            if sel.size == 0: continue
            k_take = min(targets[b], sel.size)
            if k_take>0:
                pick = rng.choice(sel, size=k_take, replace=False)
                chosen_idx_list.append(pick)
                chosen_counts[b] += pick.size
        chosen_idx = np.concatenate(chosen_idx_list) if chosen_idx_list else np.array([], dtype=int)
        # backfill by largest shortage per bucket, using remaining pool per bucket first
        remaining_by_bucket = {b: np.setdiff1d(pool_by_bucket[b], chosen_idx) for b in buckets}
        while chosen_idx.size < per_box_sim:
            # compute shortages
            shortages = [(b, max(0, targets[b] - chosen_counts[b])) for b in buckets]
            shortages.sort(key=lambda x: x[1], reverse=True)
            filled_any = False
            for b, need in shortages:
                if need <= 0: continue
                rem = remaining_by_bucket.get(b, np.array([], dtype=int))
                if rem.size == 0: continue
                take = min(need, rem.size, per_box_sim - chosen_idx.size)
                if take <= 0: continue
                pick = rng.choice(rem, size=take, replace=False)
                chosen_idx = np.concatenate([chosen_idx, pick])
                chosen_counts[b] += take
                remaining_by_bucket[b] = np.setdiff1d(rem, pick)
                filled_any = True
                if chosen_idx.size >= per_box_sim:
                    break
            if not filled_any:
                # backfill from any remaining pool
                pool_any = np.setdiff1d(idx, chosen_idx)
                if pool_any.size == 0:
                    break
                take = min(per_box_sim - chosen_idx.size, pool_any.size)
                pick = rng.choice(pool_any, size=take, replace=False)
                chosen_idx = np.concatenate([chosen_idx, pick])
                break
        # assemble rows
        if chosen_idx.size>0:
            # quick map of index->bucket for this box
            bmap = {int(ii): str(bb) for ii, bb in zip(idx.tolist(), buck.tolist())}
            for j in chosen_idx.tolist():
                d = mm_dims[j].tolist()
                ini = mm_init7[j].tolist()
                fin = mm_final7[j].tolist()
                tl = mm_tloc3[j]
                Rloc = r6_to_R_batched(mm_rloc6[j:j+1])[0]
                Ro = quat_wxyz_to_R_batched(np.asarray(ini[3:7], np.float32)[None,:])[0]
                Th = (Ro @ tl) + np.asarray(ini[:3], np.float32)
                Rh = Ro @ Rloc
                qh = _wxyz_from_R_batch(Rh[None,:,:])[0].tolist()
                sim_rows.append({
                    "object_dimensions": [float(x) for x in d],
                    "initial_object_pose": [float(x) for x in ini],
                    "final_object_pose":   [float(x) for x in fin],
                    "grasp_pose":          [float(Th[0]), float(Th[1]), float(Th[2]), float(qh[0]), float(qh[1]), float(qh[2]), float(qh[3])],
                    "collision_label":     float(mm_y[j,0]),
                    "bucket":              bmap.get(int(j), None),
                })

    # write sim JSONL
    os.makedirs(os.path.dirname(out_sim_jsonl), exist_ok=True)
    with open(out_sim_jsonl, "w") as f:
        for r in sim_rows[:total_sim]:
            f.write(json.dumps(r) + "\n")

    # reality subset: stratified sample from sim_rows per box and per bucket with same targets
    real_rows = []
    target_boxes = box_keys[:boxes_real]
    # group sim_rows by box key and bucket
    from collections import defaultdict
    box_bucket_to_rows = defaultdict(list)
    for r in sim_rows:
        k = (round(r["object_dimensions"][0],6), round(r["object_dimensions"][1],6), round(r["object_dimensions"][2],6))
        b = r.get("bucket")
        box_bucket_to_rows[(k,b)].append(r)
    for k in target_boxes:
        # per-box targets for real
        targets_real = {b: int(round(per_box_real * bucket_targets[b])) for b in buckets}
        chosen_for_box = []
        chosen_counts_rb = {b:0 for b in buckets}
        # initial take per bucket
        for b in buckets:
            pool = box_bucket_to_rows.get((k,b), [])
            if not pool: continue
            take = min(targets_real[b], len(pool))
            if take>0:
                picks = rng.choice(len(pool), size=take, replace=False)
                chosen_for_box.extend([pool[ii] for ii in picks])
                chosen_counts_rb[b] += take
        # backfill by shortage order
        while len(chosen_for_box) < per_box_real:
            shortages = [(b, max(0, targets_real[b] - chosen_counts_rb[b])) for b in buckets]
            shortages.sort(key=lambda x: x[1], reverse=True)
            filled = False
            for b, need in shortages:
                if need <= 0: continue
                pool = box_bucket_to_rows.get((k,b), [])
                # filter pool to those not already taken (by id)
                taken_ids = set(id(x) for x in chosen_for_box)
                candidates = [x for x in pool if id(x) not in taken_ids]
                if not candidates: continue
                take = min(need, len(candidates), per_box_real - len(chosen_for_box))
                picks = rng.choice(len(candidates), size=take, replace=False)
                chosen = [candidates[ii] for ii in picks]
                chosen_for_box.extend(chosen)
                chosen_counts_rb[b] += take
                filled = True
                if len(chosen_for_box) >= per_box_real:
                    break
            if not filled:
                # take from any remaining in this box
                pool_any = []
                for b2 in buckets:
                    pool_any.extend(box_bucket_to_rows.get((k,b2), []))
                taken_ids = set(id(x) for x in chosen_for_box)
                candidates = [x for x in pool_any if id(x) not in taken_ids]
                if not candidates:
                    break
                take = min(per_box_real - len(chosen_for_box), len(candidates))
                picks = rng.choice(len(candidates), size=take, replace=False)
                chosen_for_box.extend([candidates[ii] for ii in picks])
                break
        real_rows.extend(chosen_for_box[:per_box_real])

    os.makedirs(os.path.dirname(out_real_jsonl), exist_ok=True)
    with open(out_real_jsonl, "w") as f:
        for r in real_rows[:total_real]:
            f.write(json.dumps(r) + "\n")

    print(f"[subset] wrote sim={min(total_sim,len(sim_rows))} → {out_sim_jsonl}")
    print(f"[subset] wrote real={min(total_real,len(real_rows))} → {out_real_jsonl}")


def _stream_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def summarize_experiment(jsonl_path: str):
    total = 0
    dims_counts = {}
    bucket_counts = {"SMALL":0, "MEDIUM":0, "ADJACENT":0, "OPPOSITE":0}
    succ_sum = 0; succ_present=False
    # buffers for bucket compute
    init_list=[]; final_list=[]; dims_list=[]; y_list=[]
    for obj in _stream_jsonl(jsonl_path):
        total += 1
        d = obj.get("object_dimensions")
        if d is not None:
            key = (round(float(d[0]),6), round(float(d[1]),6), round(float(d[2]),6))
            dims_counts[key] = dims_counts.get(key,0) + 1
            dims_list.append(d)
        ini = obj.get("initial_object_pose")
        fin = obj.get("final_object_pose")
        if ini is not None and fin is not None:
            init_list.append(ini)
            final_list.append(fin)
        if "had_collision" in obj:
            succ_present=True; y = 1 if bool(obj["had_collision"]) else 0; succ_sum += y; y_list.append(y)
        elif "collision_label" in obj:
            succ_present=True; y = 1 if float(obj["collision_label"])>0.5 else 0; succ_sum += y; y_list.append(y)

    if init_list and final_list:
        init7 = np.asarray(init_list, np.float32)
        final7= np.asarray(final_list, np.float32)
        buck = _classify_buckets(init7, final7)
        for b in ("SMALL","MEDIUM","ADJACENT","OPPOSITE"):
            bucket_counts[b] = int((buck==b).sum())

    # dims summary
    uniq_boxes = len(dims_counts)
    top_boxes = sorted(dims_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print(f"\n[summary] {jsonl_path}")
    print(f"total={total}  unique_boxes={uniq_boxes}")
    if succ_present and total>0:
        print(f"base_rate(collision)={succ_sum/total:.3f}")
    print("bucket_counts:")
    tot_b = sum(bucket_counts.values())
    for b,c in bucket_counts.items():
        frac = (c/tot_b) if tot_b>0 else 0.0
        print(f"  {b:<8} n={c:>6} frac={frac:.3f}")
    print("top boxes (dims -> count):")
    for (dx,dy,dz), cnt in top_boxes:
        print(f"  ({dx:.6f},{dy:.6f},{dz:.6f}) -> {cnt}")


if __name__ == "__main__":
    export_test_subsets(
        memmap_dir="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test",
        out_sim_jsonl="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/sim.jsonl",
        out_real_jsonl="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/real.jsonl",
        total_sim=10000,
        total_real=1000,
        boxes_sim=10,
        boxes_real=5,
    )
    summarize_experiment(
        jsonl_path="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/sim.jsonl"
    )
    summarize_experiment(
        jsonl_path="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_test/real.jsonl"
    )
