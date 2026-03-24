# sim_deck_10k.py
import os, json, random
from collections import defaultdict, OrderedDict
from math import ceil
import numpy as np
from tqdm import tqdm

# ---------- CONFIG (edit here) ----------
RAW_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
PAIRS_TEST       = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs/pairs_test.jsonl"
OUT_PATH         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/test_deck_sim_10k.jsonl"

N_SELECT         = 10000
SEED             = 123

# Label mix controls:
#   True  → keep observed distribution from TEST (candidate pool after pick-feasibility filter)
#   False → use balanced targets (IK≈70/30, Collision≈50/50)
KEEP_IK_DIST     = False
KEEP_COL_DIST    = False
IK_TARGET_POS    = 0.85   # used when KEEP_IK_DIST=False
COL_TARGET_POS   = 0.6   # used when KEEP_COL_DIST=False

# Soft spread/caps
PICK_PEDESTAL_SOFT_QUOTA = N_SELECT // 10
PICK_PEDESTAL_SOFT_SLACK = 200         # give room so we can actually reach 10k
EDGE_CAP                 = 100000
PER_PG_CAP               = 999            # was 3
PER_G_CAP                = 999           # was 20

# Relax caps to meet class targets: when true, we ignore pick-pedestal soft cap
# while there is remaining class-bucket deficit.
RELAX_PED_CAP_DURING_DEFICIT = True

# Global-first selection settings
USE_GLOBAL_FIRST_SELECTION = True
IK_TOLERANCE = 0.02
COL_TOLERANCE = 0.02
MIN_PICK_PER_PEDESTAL = 1
REBALANCE_WITHIN_BUCKETS = True
PED_SPREAD_TOL_FRAC = 0.03
REBALANCE_MAX_SWAPS = 5000

# ---------------------------------------

def _file(raw_root, p, g):
    return os.path.join(raw_root, f"p{int(p)}", f"data_{int(g)}.json")

def _load_row(cache, raw_root, p, g, o, max_cache=4096):
    key = (int(p), int(g))
    if key in cache:
        cache.move_to_end(key, last=True)
    else:
        with open(_file(raw_root, p, g), "r") as f:
            cache[key] = json.load(f)
        if len(cache) > max_cache:
            cache.popitem(last=False)
    return cache[key][str(int(o))]

def _maybe_load_gmeta(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def _is_center(x, tol=0.1):
    return abs(float(x) - 0.5) < tol

def _hard_from_angles(ang):
    return (abs(float(ang.get("tilt_about_Xtool", 0.0))) >= 10.0) or (abs(float(ang.get("roll_about_Ytool", 0.0))) >= 5.0)

def _comfort(pick_row, place_row):
    ps = pick_row.get("local_segments", {})
    pl = place_row.get("local_segments", {})
    pk = int(ps.get("P_to_C", {}).get("ok", False)) + int(ps.get("C_to_L", {}).get("ok", False))
    lk = int(pl.get("P_to_C", {}).get("ok", False)) + int(pl.get("C_to_L", {}).get("ok", False))
    plc = 1 - int(place_row.get("endpoint_collision_at_C", {}).get("any", False))
    return pk + lk + plc  # 0..5 (higher is safer)

def _scan_candidates():
    random.seed(SEED); np.random.seed(SEED)
    gmeta = _maybe_load_gmeta(GRASPS_META_PATH)
    cache = OrderedDict()
    cands = []
    with open(PAIRS_TEST, "r") as f:
        for line in tqdm(f, desc="[scan test]"):
            rec = json.loads(line)
            p1,g1,o1 = rec["pick"]["p"],  rec["pick"]["g"],  rec["pick"]["o"]
            p2,g2,o2 = rec["place"]["p"], rec["place"]["g"], rec["place"]["o"]

            pick  = _load_row(cache, RAW_ROOT, p1,g1,o1)
            place = _load_row(cache, RAW_ROOT, p2,g2,o2)

            # Minimal viability at PICK (so run can start)
            ikp = pick.get("ik_endpoints", {})
            if not (ikp.get("C",{}).get("ok", False) and ikp.get("P",{}).get("ok", False) and ikp.get("L",{}).get("ok", False)):
                continue
            if pick.get("endpoint_collision_at_C", {}).get("any", False):
                continue

            # Labels for the *pair* (from pairing stage)
            y_ik  = int(rec["labels"]["ik"])
            y_col = int(rec["labels"]["collision"])

            # Bin tags
            dist = int(rec["bins"]["dist"])
            dori = int(rec["bins"]["dori"])
            place_ok = int(rec["bins"].get("place_ok", 1))

            # Grasp meta (by pick grasp id)
            gm   = gmeta.get(int(g1), {})
            face = str(gm.get("face", "+X"))
            u    = float(gm.get("u_frac", 0.5))
            v    = float(gm.get("v_frac", 0.5))
            ang  = gm.get("angles_deg", {})
            hard = int(_hard_from_angles(ang))
            uc   = int(_is_center(u)); vc = int(_is_center(v))

            cands.append({
                "pick":  {"p": int(p1), "g": int(g1), "o": int(o1)},
                "place": {"p": int(p2), "g": int(g2), "o": int(o2)},
                "labels": {"ik": y_ik, "col": y_col},
                "bins":   {"dist": dist, "dori": dori, "place_ok": place_ok},
                "grasp":  {"face": face, "u_frac": u, "v_frac": v, "u_center": uc, "v_center": vc, "hard": hard},
                "comfort": _comfort(pick, place),
            })
    return cands

def _derive_targets(cands):
    # Observed label rates in the candidate pool
    tot = len(cands)
    ik_pos  = sum(c["labels"]["ik"]  for c in cands)
    col_pos = sum(c["labels"]["col"] for c in cands)
    obs_ik  = ik_pos / max(1, tot)
    obs_col = col_pos / max(1, tot)
    tgt_ik  = obs_ik  if KEEP_IK_DIST  else IK_TARGET_POS
    tgt_col = obs_col if KEEP_COL_DIST else COL_TARGET_POS
    return tgt_ik, tgt_col, (obs_ik, obs_col)

# --- Preflight supply + feasible N for fixed targets ---
def _supply_report(cands):
    from collections import defaultdict
    s = defaultdict(int)
    for c in cands:
        b = (int(c["labels"]["ik"]), int(c["labels"]["col"]))
        s[b] += 1
    s_ik1  = s[(1,1)] + s[(1,0)]
    s_ik0  = s[(0,1)] + s[(0,0)]
    s_col1 = s[(1,1)] + s[(0,1)]
    s_col0 = s[(1,0)] + s[(0,0)]
    total  = s[(1,1)] + s[(1,0)] + s[(0,1)] + s[(0,0)]
    marginals = {"ik1": s_ik1, "ik0": s_ik0, "col1": s_col1, "col0": s_col0, "total": total}
    return s, marginals

def _feasible_N_for_targets(cands, tgt_ik, tgt_col):
    s, m = _supply_report(cands)
    eps = 1e-9
    N_by_ik1  = int(np.floor(m["ik1"]  / max(tgt_ik, eps)))
    N_by_ik0  = int(np.floor(m["ik0"]  / max(1.0 - tgt_ik, eps)))
    N_by_col1 = int(np.floor(m["col1"] / max(tgt_col, eps)))
    N_by_col0 = int(np.floor(m["col0"] / max(1.0 - tgt_col, eps)))
    N_by_tot  = int(m["total"])
    N_max = min(N_by_ik1, N_by_ik0, N_by_col1, N_by_col0, N_by_tot)
    return N_max, s, m

def _print_pick_supply(cands):
    from collections import defaultdict
    by_ped = defaultdict(int)
    by_ped_bucket = defaultdict(lambda: { (1,1):0,(1,0):0,(0,1):0,(0,0):0 })
    for c in cands:
        p = int(c["pick"]["p"])
        b = (int(c["labels"]["ik"]), int(c["labels"]["col"]))
        by_ped[p] += 1
        by_ped_bucket[p][b] += 1
    ped_list = sorted(by_ped.items())
    print("[supply] eligible pick candidates by pedestal:", dict(ped_list))
    zero_peds = [p for p,count in ped_list if count == 0]
    if zero_peds:
        print("[supply] WARNING: zero eligible pick supply for pedestals:", zero_peds)
    # brief joint-bucket view for top pedestals
    sample = {p: by_ped_bucket[p] for p,_ in ped_list[:10]}
    print("[supply] sample per-ped joint buckets:", sample)

def _group_by_cell(cands):
    cells = defaultdict(list)
    for c in cands:
        key = (c["bins"]["dist"], c["bins"]["dori"])  # 16 cells
        cells[key].append(c)
    # randomize within cell ONLY (no comfort bias → keeps col=1 available)
    rng = random.Random(SEED)
    for k, lst in cells.items():
        rng.shuffle(lst)
    return cells

def _global_first_select(cands, tgt_ik, tgt_col, N_target, seed=SEED):
    from collections import defaultdict
    rng = random.Random(seed)
    # Partition by joint buckets
    B = {(1,1): [], (1,0): [], (0,1): [], (0,0): []}
    for r in cands:
        b = (int(r["labels"]["ik"]), int(r["labels"]["col"]))
        B[b].append(r)
    for b in B: rng.shuffle(B[b])

    supply = {b: len(B[b]) for b in B}

    # Compute quotas that satisfy marginals exactly within supply bounds
    def compute_quota(N):
        want_ik  = int(round(tgt_ik  * N))
        want_col = int(round(tgt_col * N))
        # x11 bounds from constraints
        lo = 0
        hi = min(supply[(1,1)], want_ik, want_col)
        lo = max(lo, want_ik - supply[(1,0)])
        lo = max(lo, want_col - supply[(0,1)])
        lo = max(lo, want_ik + want_col - N)  # from x00 >= 0
        hi = min(hi, N - (want_ik + want_col) + supply[(0,0)])  # from x00 <= s00
        if lo > hi:
            return None
        x11 = lo  # choose minimal IK-boosting solution to not overshoot IK
        x10 = want_ik  - x11
        x01 = want_col - x11
        x00 = N - (x11 + x10 + x01)
        if x10 < 0 or x01 < 0 or x00 < 0:
            return None
        if x10 > supply[(1,0)] or x01 > supply[(0,1)] or x00 > supply[(0,0)]:
            return None
        return {(1,1): x11, (1,0): x10, (0,1): x01, (0,0): x00}

    quotas = compute_quota(N_target)
    if quotas is None:
        # Try relaxing within tolerance by shrinking N until feasible
        N_try = N_target
        while N_try > 0 and quotas is None:
            N_try -= 100
            quotas = compute_quota(N_try)
        if quotas is None:
            # Fallback to proportional quotas clipped by supply
            want_ik  = int(round(tgt_ik  * N_target))
            want_col = int(round(tgt_col * N_target))
            n11 = min(int(round(tgt_ik * tgt_col * N_target)), supply[(1,1)])
            n10 = min(want_ik  - n11, supply[(1,0)])
            n01 = min(want_col - n11, supply[(0,1)])
            n00 = min(N_target - (n11 + n10 + n01), supply[(0,0)])
            quotas = {(1,1): max(0,n11), (1,0): max(0,n10), (0,1): max(0,n01), (0,0): max(0,n00)}

    # Materialize selection according to quotas, ensuring pedestal coverage first
    sel = []
    selected_keys = set()
    ped_pick = defaultdict(int)
    ped_place= defaultdict(int)
    edges = defaultdict(int)

    def key_of(r):
        return (r["pick"]["p"], r["pick"]["g"], r["pick"]["o"], r["place"]["p"], r["place"]["g"], r["place"]["o"]) 

    # Build per-pedestal bucket pools for seeding
    pedestals = sorted({int(r["pick"]["p"]) for r in cands})
    items_by_ped_bucket = defaultdict(list)
    for b in B:
        for r in B[b]:
            items_by_ped_bucket[(int(r["pick"]["p"]), b)].append(r)
    for k in list(items_by_ped_bucket.keys()):
        random.Random(seed).shuffle(items_by_ped_bucket[k])

    # Seed: at least MIN_PICK_PER_PEDESTAL per available pedestal (best effort within quotas)
    for p in pedestals:
        # Try buckets in an order that preserves quotas but favors negatives when possible
        bucket_order = [(0,1), (0,0), (1,0), (1,1)]
        while ped_pick[p] < MIN_PICK_PER_PEDESTAL and len(sel) < N_target:
            picked = False
            for bb in bucket_order:
                if quotas.get(bb, 0) <= 0:
                    continue
                lst = items_by_ped_bucket.get((p, bb), [])
                while lst:
                    r = lst.pop()
                    k = key_of(r)
                    if k in selected_keys:
                        continue
                    # take r for seeding
                    sel.append(r)
                    selected_keys.add(k)
                    quotas[bb] -= 1
                    p1 = int(r["pick"]["p"]); p2 = int(r["place"]["p"])
                    ped_pick[p1]  += 1
                    ped_place[p2] += 1
                    edges[(p1,p2)] += 1
                    picked = True
                    break
                if picked:
                    break
            if not picked:
                break  # no feasible item for this pedestal under remaining quotas

    # Fill remaining quotas from bucket pools
    for b,count in quotas.items():
        took = 0
        pool = B[b]
        while took < count and pool and len(sel) < N_target:
            r = pool.pop()
            k = key_of(r)
            if k in selected_keys:
                continue
            sel.append(r)
            selected_keys.add(k)
            p1,g1 = r["pick"]["p"], r["pick"]["g"]
            p2    = r["place"]["p"]
            ped_pick[p1]  += 1
            ped_place[p2] += 1
            edges[(p1,p2)] += 1
            took += 1

    # Optional within-bucket pedestal smoothing without changing quotas
    if REBALANCE_WITHIN_BUCKETS and len(sel) == N_target:
        # Build remaining pool per (pedestal, bucket) from current B
        remaining_by_ped_bucket = defaultdict(list)
        for b in B:
            for r in B[b]:
                remaining_by_ped_bucket[(int(r["pick"]["p"]), b)].append(r)
        for k in list(remaining_by_ped_bucket.keys()):
            rng.shuffle(remaining_by_ped_bucket[k])

        # Selected indices per bucket and pedestal
        def bucket_of(rec):
            return (int(rec["labels"]["ik"]), int(rec["labels"]["col"]))

        indices_by_ped_bucket = defaultdict(list)
        for idx, r in enumerate(sel):
            indices_by_ped_bucket[(int(r["pick"]["p"]), bucket_of(r))].append(idx)

        ped_ids = sorted({int(r["pick"]["p"]) for r in cands})
        ideal = float(len(sel)) / max(1, len(ped_ids))
        tol = int(round(PED_SPREAD_TOL_FRAC * len(sel)))
        low = int(max(0, ideal - tol))
        high = int(ideal + tol)

        # Build selected key set for uniqueness
        def key_of(r):
            return (r["pick"]["p"], r["pick"]["g"], r["pick"]["o"], r["place"]["p"], r["place"]["g"], r["place"]["o"]) 
        selected_keys = set(key_of(r) for r in sel)

        swaps = 0
        improved = True
        while improved and swaps < REBALANCE_MAX_SWAPS:
            improved = False
            # Identify over/under pedestals
            over = [p for p,c in sorted(ped_pick.items(), key=lambda x:-x[1]) if c > high]
            under= [p for p,c in sorted(ped_pick.items(), key=lambda x:x[1])  if c < low]
            if not over or not under:
                break
            for p_over in over:
                if swaps >= REBALANCE_MAX_SWAPS:
                    break
                for p_under in under:
                    if ped_pick[p_under] >= low:
                        continue
                    # Try to swap within any bucket
                    for b in [(1,1),(1,0),(0,1),(0,0)]:
                        idx_list = indices_by_ped_bucket.get((p_over, b), [])
                        pool = remaining_by_ped_bucket.get((p_under, b), [])
                        if not idx_list or not pool:
                            continue
                        # take one candidate from pool that keeps uniqueness
                        replacement = None
                        while pool and replacement is None:
                            cand = pool.pop()
                            k = key_of(cand)
                            if k in selected_keys:
                                continue
                            replacement = cand
                        if replacement is None:
                            continue
                        victim_idx = idx_list.pop()
                        victim = sel[victim_idx]
                        # swap
                        sel[victim_idx] = replacement
                        selected_keys.add(key_of(replacement))
                        selected_keys.discard(key_of(victim))
                        # update counts/maps
                        ped_pick[p_over] -= 1
                        ped_pick[p_under] += 1
                        p2_old = int(victim["place"]["p"]) ; p2_new = int(replacement["place"]["p"]) 
                        edges[(p_over, p2_old)] = max(0, edges.get((p_over,p2_old),0) - 1)
                        edges[(p_under, p2_new)] = edges.get((p_under,p2_new),0) + 1
                        # victim goes back to remaining pool
                        remaining_by_ped_bucket[(p_over, b)].append(victim)
                        rng.shuffle(remaining_by_ped_bucket[(p_over, b)])
                        # replacement index belongs to p_under now
                        indices_by_ped_bucket[(p_under, b)].append(victim_idx)
                        swaps += 1
                        improved = True
                        break
                    if improved:
                        break
                if improved and ped_pick[p_over] <= high:
                    continue

    stats = {"ped_pick": dict(ped_pick), "ped_place": dict(ped_place), "edges": dict(edges)}
    return sel, stats

def _select(cells, tgt_ik, tgt_col, N_override=None):
    """
    Supply-aware joint-bucket selection that enforces BOTH targets
    and fills exactly N_SELECT unless the pool truly lacks supply.
    """
    from collections import defaultdict, deque
    rng   = random.Random(SEED)
    keys  = sorted(cells.keys())
    N     = int(N_override if N_override is not None else N_SELECT)

    # 1) Build per-cell bucket queues and measure global supply
    B = {}
    supply = {(1,1): 0, (1,0): 0, (0,1): 0, (0,0): 0}
    for k in keys:
        lst = list(cells[k])
        rng.shuffle(lst)
        buckets = {(1,1): deque(), (1,0): deque(), (0,1): deque(), (0,0): deque()}
        for r in lst:
            b = (int(r["labels"]["ik"]), int(r["labels"]["col"]))
            buckets[b].append(r)
            supply[b] += 1
        B[k] = buckets

    # 2) Desired counts and clip to supply
    want_ik  = int(round(tgt_ik  * N))
    want_col = int(round(tgt_col * N))
    n11 = int(round(tgt_ik * tgt_col * N))
    n10 = want_ik  - n11
    n01 = want_col - n11
    n00 = N - (n11 + n10 + n01)
    need = {(1,1): n11, (1,0): n10, (0,1): n01, (0,0): n00}

    # Clip by available supply
    for b in need:
        if need[b] > supply[b]:
            need[b] = supply[b]
    filled_target = sum(need.values())

    # If we clipped, we may still be < N; fill remaining from buckets with spare supply,
    # always choosing the bucket that best reduces (IK,COL) deficit.
    def planned_counts_from_need(nd):
        # planned counts achieved so far relative to desired (nXX - nd[XX])
        ik = (n11 - nd[(1,1)]) + (n10 - nd[(1,0)])
        co = (n11 - nd[(1,1)]) + (n01 - nd[(0,1)])
        return ik, co

    def choose_bucket_reduce_error_realloc(nd, b_from):
        # Try reallocating one unit from b_from to bb to reduce squared error on (IK, COL)
        ik_now, col_now = planned_counts_from_need(nd)
        d_ik  = want_ik  - ik_now
        d_col = want_col - col_now
        best_bb = None
        best_err = None
        for bb in need.keys():
            if bb == b_from:
                continue
            avail = sum(len(B[k][bb]) for k in keys)
            if avail <= 0:
                continue
            # simulate reallocation: nd' = nd with nd[b_from]-=1, nd[bb]+=1
            v_from_ik  = 1 if b_from in [(1,1),(1,0)] else 0
            v_from_col = 1 if b_from in [(1,1),(0,1)] else 0
            v_bb_ik    = 1 if bb     in [(1,1),(1,0)] else 0
            v_bb_col   = 1 if bb     in [(1,1),(0,1)] else 0
            # planned counts after reallocation change by (v_from - v_bb)
            ik_new  = ik_now  + (v_from_ik  - v_bb_ik)
            col_new = col_now + (v_from_col - v_bb_col)
            err = (want_ik - ik_new)**2 + (want_col - col_new)**2
            # do not choose moves that worsen both IK and COL deficits simultaneously
            if (abs(want_ik - ik_new) >= abs(d_ik)) and (abs(want_col - col_new) >= abs(d_col)):
                continue
            if (best_bb is None) or (err < best_err):
                best_bb = bb; best_err = err
        return best_bb

    # If clipped total < N, expand needs toward N using the heuristic above
    # expansion not needed when initial targets already sum to N

    # 3) Selection loop: try to realize 'need' using caps
    ped_pick = defaultdict(int)
    ped_place= defaultdict(int)
    edge_use = defaultdict(int)
    per_pg   = defaultdict(int)
    per_g    = defaultdict(int)

    def ok_caps(r):
        p1,g1,o1 = r["pick"]["p"], r["pick"]["g"], r["pick"]["o"]
        p2       = r["place"]["p"]
        if per_pg[(p1,g1)] >= PER_PG_CAP:                                       return False
        if per_g[g1]        >= PER_G_CAP:                                       return False
        if edge_use[(p1,p2)]>= EDGE_CAP:                                        return False
        # Relax pick-pedestal soft cap when we still have target deficits
        if not RELAX_PED_CAP_DURING_DEFICIT or (sum(need.values()) <= 0):
            if ped_pick[p1] > (PICK_PEDESTAL_SOFT_QUOTA + PICK_PEDESTAL_SOFT_SLACK):
                return False
        return True

    def pull_from_bucket(b):
        # Prefer filling deficits from minority classes first: when b is a positive-IK bucket
        # and IK is already above target, try alternate buckets; same for COL.
        progressed = False
        for k in keys:
            pool = B[k][b]
            while pool:
                r = pool[0]
                if not ok_caps(r):
                    pool.popleft()
                    continue
                # take
                sel.append(r)
                p1,g1 = r["pick"]["p"], r["pick"]["g"]
                p2    = r["place"]["p"]
                per_pg[(p1,g1)] += 1
                per_g[g1]       += 1
                ped_pick[p1]    += 1
                ped_place[p2]   += 1
                edge_use[(p1,p2)] += 1
                need[b] -= 1
                pool.popleft()
                progressed = True
                break
            if progressed:
                break
        return progressed

    # 3) Selection loop: satisfy bucket 'need' first; stop only when all needs → 0
    sel = []
    stall_passes = 0
    def total_need(nd): return sum(nd.values())

    while (total_need(need) > 0) and (len(sel) < N) and (stall_passes < 10):
        progressed = False
        # always try the bucket with the LARGEST remaining need first; prefer minority buckets
        for b,_rem in sorted(need.items(), key=lambda kv: -kv[1]):
            if need[b] <= 0:
                continue
            if pull_from_bucket(b):
                progressed = True
                break  # re-evaluate needs after each success
        if not progressed:
            # try to reallocate one unit of the hardest unmet need to a feasible bucket
            # (caps-aware fallback; keeps IK/COL close to target)
            b_hard, _ = max(need.items(), key=lambda kv: kv[1])
            # try to reallocate toward reducing squared error on (IK, COL)
            bb = choose_bucket_reduce_error_realloc(need, b_hard)
            if bb is not None and need[b_hard] > 0:
                need[b_hard] -= 1
                need[bb]     += 1
                stall_passes = 0
            else:
                stall_passes += 1

    # Do not pad beyond computed needs; return what we achieved toward N

    stats = {
        "ped_pick": dict(ped_pick),
        "ped_place": dict(ped_place),
        "edges": dict(edge_use),
    }
    return sel, stats

def _write_deck(path, sel):
    with open(path, "w") as f:
        for i,c in enumerate(sel, 1):
            out = {
                "id": f"S{str(i).zfill(5)}",
                "pick":  c["pick"],
                "place": c["place"],
                "bins":  c["bins"],
                "labels": c["labels"],
                "grasp": {"face": c["grasp"]["face"],
                          "u_frac": c["grasp"]["u_frac"],
                          "v_frac": c["grasp"]["v_frac"],
                          "u_center": c["grasp"]["u_center"],
                          "v_center": c["grasp"]["v_center"],
                          "hard": c["grasp"]["hard"]},
                "comfort": c["comfort"],
            }
            f.write(json.dumps(out) + "\n")

def _print_report(sel, targets, observed, ped_stats):
    tgt_ik, tgt_col = targets
    obs_ik, obs_col = observed
    n = len(sel)
    ik_pos  = sum(s["labels"]["ik"]  for s in sel)
    col_pos = sum(s["labels"]["col"] for s in sel)
    print(f"Selected: {n}")
    print(f"IK  pos={ik_pos} ({ik_pos/n:.3f})  target={tgt_ik:.3f}  observed_pool={obs_ik:.3f}")
    print(f"COL pos={col_pos} ({col_pos/n:.3f})  target={tgt_col:.3f} observed_pool={obs_col:.3f}")
    # coverage by cells
    by_cell = defaultdict(int)
    for s in sel: by_cell[(s["bins"]["dist"], s["bins"]["dori"])] += 1
    print("Cell coverage (dist,dori → count):", dict(sorted(by_cell.items())))
    # pedestal spread (pick)
    print("Pick pedestal counts:", dict(sorted(ped_stats["ped_pick"].items())))
    # small face/center/hard table
    face_tab = defaultdict(int)
    for s in sel:
        key = (s["grasp"]["face"], s["grasp"]["u_center"], s["grasp"]["v_center"], s["grasp"]["hard"])
        face_tab[key] += 1
    sample_face = dict(sorted(face_tab.items())[:12])
    print("Sample (face,u_center,v_center,hard) counts:", sample_face)

# Add near the other helpers
def _sanity_check_output(path=OUT_PATH):
    from collections import Counter, OrderedDict
    cache = OrderedDict()
    seen  = set()
    n = ik_pos = col_pos = 0
    by_cell = Counter(); ped_pick = Counter(); edge_use = Counter()
    bad_pick_feas = 0
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line); n += 1
            key = (rec["pick"]["p"],rec["pick"]["g"],rec["pick"]["o"],
                   rec["place"]["p"],rec["place"]["g"],rec["place"]["o"])
            if key in seen: continue  # skip dup count; report below
            seen.add(key)
            by_cell[(rec["bins"]["dist"], rec["bins"]["dori"])] += 1
            ped_pick[rec["pick"]["p"]] += 1
            edge_use[(rec["pick"]["p"], rec["place"]["p"])] += 1
            ik_pos  += int(rec["labels"]["ik"])
            col_pos += int(rec["labels"]["col"])
            # recheck pick feasibility so every trial can start
            pick = _load_row(cache, RAW_ROOT, rec["pick"]["p"], rec["pick"]["g"], rec["pick"]["o"])
            okC = pick.get("ik_endpoints",{}).get("C",{}).get("ok",False)
            okP = pick.get("ik_endpoints",{}).get("P",{}).get("ok",False)
            okL = pick.get("ik_endpoints",{}).get("L",{}).get("ok",False)
            col = pick.get("endpoint_collision_at_C",{}).get("any",False)
            if not (okC and okP and okL) or col: bad_pick_feas += 1

    print(f"[sanity] lines={n} unique={len(seen)} dups={n-len(seen)}")
    print(f"[sanity] IK pos={ik_pos} ({ik_pos/max(1,n):.3f})  COL pos={col_pos} ({col_pos/max(1,n):.3f})")
    print(f"[sanity] cells hit={len(by_cell)}/16 → {dict(sorted(by_cell.items()))}")
    print(f"[sanity] pick pedestal spread → {dict(sorted(ped_pick.items()))}")
    print(f"[sanity] edge caps (top 10) → {sorted(edge_use.items(), key=lambda x: -x[1])[:10]}")
    print(f"[sanity] bad pick-feasibility (should be 0): {bad_pick_feas}")

def _read_deck(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def _post_balance_deck(src_path, dst_path, ik_target=0.70, col_target=0.50, seed=SEED):
    """
    2-stage post-balance:
      (A) IK per cell to target=ik_target by only REMOVING items, maximizing kept count.
          For a cell with P positives, N negatives and target t:
            K <= min(floor(P/t), floor(N/(1-t))). Keep exactly t*K positives and (1-t)*K negatives.
          If a cell cannot achieve t exactly (e.g., P or N is 0), we keep whatever exists (best effort).
      (B) Collision per cell to 50/50 by REMOVING down to 2*min(npos, nneg) in each cell.
    """
    from collections import defaultdict
    rng = random.Random(seed)

    # --- load deck and group by cell ---
    items = _read_deck(src_path)
    cells = defaultdict(list)
    for r in items:
        cells[(int(r["bins"]["dist"]), int(r["bins"]["dori"]))].append(r)

    # --- (A) IK per-cell balancing to exact t where possible ---
    t = float(ik_target)
    kept_after_ik = []
    for key, lst in cells.items():
        P = [r for r in lst if int(r["labels"]["ik"]) == 1]
        N = [r for r in lst if int(r["labels"]["ik"]) == 0]
        rng.shuffle(P); rng.shuffle(N)
        if t <= 0.0 or t >= 1.0 or (len(P) == 0 and len(N) == 0):
            # degenerate; keep nothing
            continue
        # maximal K that permits exact ratio t with only removals
        K1 = int(np.floor(len(P) / max(t, 1e-9)))              # bound by positives
        K2 = int(np.floor(len(N) / max(1.0 - t, 1e-9)))        # bound by negatives
        K = min(K1, K2)
        if K <= 0:
            # can't realize target; keep what exists (best effort)
            kept_after_ik.extend(lst)
            continue
        keep_pos = int(round(t * K))
        keep_neg = K - keep_pos
        kept_after_ik.extend(P[:keep_pos] + N[:keep_neg])

    # --- (B) Collision per-cell balancing to exact 50/50 where possible ---
    cells2 = defaultdict(list)
    for r in kept_after_ik:
        cells2[(int(r["bins"]["dist"]), int(r["bins"]["dori"]))].append(r)

    balanced = []
    for key, lst in cells2.items():
        pos = [r for r in lst if int(r["labels"]["col"]) == 1]
        neg = [r for r in lst if int(r["labels"]["col"]) == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        k = min(len(pos), len(neg))
        if k == 0:
            # no way to make 50/50; keep as-is
            balanced.extend(lst)
            continue
        # exact 50/50 per-cell
        balanced.extend(pos[:k] + neg[:k])

    # --- write out and print a tiny report ---
    with open(dst_path, "w") as f:
        for r in balanced:
            f.write(json.dumps(r) + "\n")

    n = len(balanced)
    ikp = sum(int(r["labels"]["ik"])  for r in balanced)
    colp= sum(int(r["labels"]["col"]) for r in balanced)
    print(f"[post-balance] wrote {n} → IK={ikp/n if n else 0:.3f}  COL={colp/n if n else 0:.3f}  (targets {ik_target:.2f}, {col_target:.2f})")



def main():
    cands = _scan_candidates()
    tgt_ik, tgt_col, observed = _derive_targets(cands)
    N_max, supply, marg = _feasible_N_for_targets(cands, tgt_ik, tgt_col)
    N_target = min(int(N_SELECT), int(N_max))
    print(f"[supply] total={marg['total']} ik1={marg['ik1']} ik0={marg['ik0']} col1={marg['col1']} col0={marg['col0']}")
    print(f"[targets] IK={tgt_ik:.2f} COL={tgt_col:.2f}  → feasible_N≤{N_max}  using N={N_target}")
    _print_pick_supply(cands)
    if USE_GLOBAL_FIRST_SELECTION:
        sel, ped_stats = _global_first_select(cands, tgt_ik, tgt_col, N_target, seed=SEED)
    else:
        cells = _group_by_cell(cands)
        sel, ped_stats = _select(cells, tgt_ik, tgt_col, N_override=N_target)
    _write_deck(OUT_PATH, sel)
    _print_report(sel, (tgt_ik, tgt_col), observed, ped_stats)
    _sanity_check_output(OUT_PATH)
    print(f"Wrote deck → {OUT_PATH}")

    # Optional per-cell post-balance disabled to preserve global mix; keep function available

if __name__ == "__main__":
    main()
