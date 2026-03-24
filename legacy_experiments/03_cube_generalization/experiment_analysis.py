# ======== minimal experiment evaluator (p_success -> p_collision) ========

import json, os, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

# --- EDIT THESE TWO PATHS ONLY ---
FILE_A = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/experiment_results_test_data.indexed.jsonl"     # your new model run
FILE_B = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/experiment_results_v4.jsonl"     # your GPD model run (or leave None)
FILE_C = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/new_corners_only_recollect.jsonl"    # your new model run
# ---------------------------------

FILTER_IK_FAIL = True   # set False to include IK_fail rows

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rows.append(json.loads(line))
    return rows

def to_arrays(rows):
    """Return (y, p_collision) arrays using your rule:
       y = 1 if collision_counter>0 else 0
       p_collision = 1 - prediction_score   (since prediction_score is p_success)
    """
    y, p = [], []
    for r in rows:
        if FILTER_IK_FAIL and r.get("reason") == "IK_fail":
            continue
        if r.get("grasp") is not True:
            continue

        if "collision_counter" not in r or "prediction_score" not in r:
            continue

        y.append(1 if (r["collision_counter"] or 0) > 0 else 0)
        p.append(1.0 - float(r["prediction_score"]))  # invert once: success -> collision
    return np.asarray(y, int), np.asarray(p, float)

def sweep_thresholds(y, p):
    ths = np.linspace(0.0, 1.0, 1001)  # finer sweep is ok
    best_bal = (-1.0, 0.5)  # (score, tau)
    best_f1  = (-1.0, 0.5)
    best_acc = (-1.0, 0.5)
    for t in ths:
        yhat = (p >= t).astype(int)
        bal  = balanced_accuracy_score(y, yhat)
        f1   = f1_score(y, yhat, zero_division=0)
        acc  = (yhat == y).mean()

        if bal > best_bal[0]: best_bal = (bal, t)
        if f1  > best_f1[0]:  best_f1  = (f1,  t)
        if acc > best_acc[0]: best_acc = (acc, t)
    return best_bal, best_f1, best_acc

def confusion(y, p, tau):
    yhat = (p >= tau).astype(int)
    tp = int(((y==1)&(yhat==1)).sum())
    fp = int(((y==0)&(yhat==1)).sum())
    tn = int(((y==0)&(yhat==0)).sum())
    fn = int(((y==1)&(yhat==0)).sum())
    acc = (tp+tn)/len(y)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1, tau=tau)

def eval_file(path, title):
    if not path or not os.path.exists(path):
        print(f"\n[{title}] file not found: {path}")
        return
    rows = load_rows(path)
    y, p = to_arrays(rows)
    if len(y) == 0:
        print(f"\n[{title}] no valid rows after filtering")
        return
    roc = roc_auc_score(y, p)
    pr  = average_precision_score(y, p)
    (bal, tau_bal), (f1s, tau_f1), (accs, tau_acc) = sweep_thresholds(y, p)
    m_bal = confusion(y, p, tau_bal)
    m_acc = confusion(y, p, tau_acc)

    print(f"\n[{title}] rows={len(y)}  pos_rate={y.mean():.3f}")
    print(f"  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}")
    print(f"  Best τ by BalancedAcc: τ={tau_bal:.2f}  BalAcc={bal:.3f}")
    print(f"  Best τ by F1:          τ={tau_f1:.2f}  F1={f1s:.3f}")
    print(f"  Best τ by Accuracy:    τ={tau_acc:.2f}  Acc={accs:.3f}")
    print(f"  Confusion @ τ_bal: [tp fp tn fn] = [{m_bal['tp']} {m_bal['fp']} {m_bal['tn']} {m_bal['fn']}]")
    print(f"    Acc={m_bal['acc']:.3f}  Prec={m_bal['prec']:.3f}  Rec={m_bal['rec']:.3f}  F1={m_bal['f1']:.3f}")
    print(f"  Confusion @ τ_acc: [tp fp tn fn] = [{m_acc['tp']} {m_acc['fp']} {m_acc['tn']} {m_acc['fn']}]")
    print(f"    Acc={m_acc['acc']:.3f}  Prec={m_acc['prec']:.3f}  Rec={m_acc['rec']:.3f}  F1={m_acc['f1']:.3f}")

def main():
    eval_file(FILE_A, "INPUT A")
    if FILE_B:
        eval_file(FILE_B, "INPUT B")
    if FILE_C:
        eval_file(FILE_C, "INPUT C")

if __name__ == "__main__":
    main()
