# file: train_pickplace.py
import os, json, math, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from placement_quality.path_simulation.model_training.precompute import precompute_split

# ---- paths & toggles (edit these) ----
RAW_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/data_collection/raw_data"
GRASPS_META_PATH = "/home/chris/Chris/placement_ws/src/grasps_meta_data.json"
PAIRS_DIR        = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/pairs"
PAIRS_TRAIN      = os.path.join(PAIRS_DIR, "pairs_train.jsonl")
PAIRS_VAL        = os.path.join(PAIRS_DIR, "pairs_val.jsonl")
PAIRS_TEST       = os.path.join(PAIRS_DIR, "pairs_test.jsonl")


USE_CORNERS = True     # match what you want to use
USE_META    = True
USE_DELTA   = True
USE_PRECOMPUTED = True  # set True after running precompute.py

# New ablation toggle
USE_TRANSPORT = False    # express grasp in transport frame

# Optional automation
AUTO_PRECOMPUTE = True   # run precompute for train/val/test before training
AUTO_EVAL_DECK = True    # run deck evaluation after training

def _compose_precomp_root():
    base = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/precomputed"
    # Mirror precompute.py: create a subfolder per case
    sub = []
    sub.append("meta" if USE_META else "nometa")
    sub.append("delta" if USE_DELTA else "abs")
    sub.append("corners" if USE_CORNERS else "nocorners")
    sub.append("transport" if USE_TRANSPORT else "world")
    return os.path.join(base, "-".join(sub))

PRECOMP_ROOT = _compose_precomp_root()
# ---- training hyperparams ----
EPOCHS      = 150
PATIENCE    = 10  # allow more time for calibration
BATCH_SIZE  = 4096
LR          = 1e-3
WD          = 3e-4
DROPOUT     = 0.10
HIDDEN      = 64
NUM_WORKERS = 16
CLIP_NORM   = 2.0
OUT_DIR     = "/home/chris/Chris/placement_ws/src/data/box_simulation/v7/training_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- class weights (from your counts) ----
# IK: pos=3,016,709  neg=358,892
IK_POS, IK_NEG = 3016709, 358892
IK_POS_W = 0.5 * (IK_POS + IK_NEG) / IK_POS   # ~0.56
IK_NEG_W = 0.5 * (IK_POS + IK_NEG) / IK_NEG   # ~4.70

# Collision: pos=1,594,739  neg=1,780,862  (near-balanced)
COL_POS, COL_NEG = 1594739, 1780862
COL_POS_W = 0.5 * (COL_POS + COL_NEG) / COL_POS
COL_NEG_W = 0.5 * (COL_POS + COL_NEG) / COL_NEG

# ---- import your dataset & model ----
from placement_quality.path_simulation.model_training.model import PickPlaceDataset, PickPlaceFeasibilityNet

def bce_with_class_weights(logits, targets, pos_w, neg_w, eps=0.05):
    """Per-sample weighted BCE (keeps it simple & explicit)."""
    # logits, targets: [B,1]
    t = targets * (1.0 - eps) + 0.5 * eps
    bce = nn.functional.binary_cross_entropy_with_logits(logits, t, reduction='none')
    w = targets * pos_w + (1.0 - targets) * neg_w
    return (bce * w).mean()

def batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out

@torch.no_grad()
def evaluate(model, loader, device, use_meta, use_corners, ik_w=(1.0,1.0), col_w=(1.0,1.0), weighted=False):
    model.eval()
    n, loss_ik_sum, loss_col_sum = 0, 0.0, 0.0
    corr_ik = corr_col = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        li, lc = model(
            batch["grasp_Oi"].float(),
            batch["objW_pick"].float(),
            batch["objW_place"].float(),
            meta=batch.get("meta", None).float() if use_meta and "meta" in batch else None,
            corners_f=batch.get("corners_f", None).float() if use_corners and "corners_f" in batch else None,
        )
        yik  = batch["y_ik"]
        ycol = batch["y_col"]

        if weighted:
            loss_ik  = bce_with_class_weights(li, yik,  pos_w=ik_w[0],  neg_w=ik_w[1])
            loss_col = bce_with_class_weights(lc, ycol, pos_w=col_w[0], neg_w=col_w[1])
        else:
            loss_ik  = nn.functional.binary_cross_entropy_with_logits(li, yik)
            loss_col = nn.functional.binary_cross_entropy_with_logits(lc, ycol)

        # accuracies at 0.5
        pred_ik  = (li.sigmoid()  >= 0.5).float()
        pred_col = (lc.sigmoid()  >= 0.5).float()
        corr_ik  += (pred_ik  == yik).sum().item()
        corr_col += (pred_col == ycol).sum().item()

        bsz = yik.shape[0]
        n += bsz
        loss_ik_sum  += loss_ik.item()  * bsz
        loss_col_sum += loss_col.item() * bsz

    return {
        "loss_ik":  loss_ik_sum / n,
        "loss_col": loss_col_sum / n,
        "acc_ik":   corr_ik  / n,
        "acc_col":  corr_col / n,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(SEED)
    dl_gen = torch.Generator(device="cpu")
    dl_gen.manual_seed(SEED)

    # Optionally precompute memmaps to match current toggles
    if USE_PRECOMPUTED and AUTO_PRECOMPUTE:
        os.makedirs(PRECOMP_ROOT, exist_ok=True)
        print(f"[precompute] Writing precomputed features to {PRECOMP_ROOT} …")
        precompute_split("train", PAIRS_TRAIN, PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)
        precompute_split("val",   PAIRS_VAL,   PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)
        precompute_split("test",  PAIRS_TEST,  PRECOMP_ROOT, RAW_ROOT, GRASPS_META_PATH, USE_META, USE_CORNERS, USE_DELTA)

    # datasets / loaders
    ds_train = PickPlaceDataset(PAIRS_TRAIN, RAW_ROOT, GRASPS_META_PATH,
                                use_corners=USE_CORNERS, use_meta=USE_META, use_delta=USE_DELTA,
                                precomp_dir=os.path.join(PRECOMP_ROOT, "train") if USE_PRECOMPUTED else None)
    ds_val   = PickPlaceDataset(PAIRS_VAL,   RAW_ROOT, GRASPS_META_PATH,
                                use_corners=USE_CORNERS, use_meta=USE_META, use_delta=USE_DELTA,
                                precomp_dir=os.path.join(PRECOMP_ROOT, "val") if USE_PRECOMPUTED else None)
    ds_test  = PickPlaceDataset(PAIRS_TEST,  RAW_ROOT, GRASPS_META_PATH,
                                use_corners=USE_CORNERS, use_meta=USE_META, use_delta=USE_DELTA,
                                precomp_dir=os.path.join(PRECOMP_ROOT, "test") if USE_PRECOMPUTED else None)

    PERSIST = NUM_WORKERS > 0
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                          generator=dl_gen, persistent_workers=PERSIST)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                          persistent_workers=PERSIST)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                          persistent_workers=PERSIST)

    # model / opt — make flags match actual dataset features to avoid shape mismatch
    USE_META_MODEL = USE_META and hasattr(ds_train, "view") and ("meta" in ds_train.view)
    USE_CORNERS_MODEL = USE_CORNERS and hasattr(ds_train, "view") and ("corners_f" in ds_train.view)

    model = PickPlaceFeasibilityNet(use_meta=USE_META_MODEL, use_corners=USE_CORNERS_MODEL,
                                    hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # Warmup + cosine decay
    WARMUP_EPOCHS = 5
    def lr_lambda(epoch):
        # epoch is 0-indexed here
        if epoch < WARMUP_EPOCHS:
            return max(1e-3, float(epoch + 1) / float(WARMUP_EPOCHS))
        t = epoch - WARMUP_EPOCHS
        tmax = max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * t / tmax))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    # logs for plotting
    hist = {"tr_loss_ik":[], "tr_loss_col":[], "tr_acc_ik":[], "tr_acc_col":[],
            "va_loss_ik":[], "va_loss_col":[], "va_acc_ik":[], "va_acc_col":[]}

    best_val_total = math.inf
    best_val_ik = math.inf
    best_val_col = math.inf
    no_improve = 0
    ckpt_path = os.path.join(OUT_DIR, "best.pt")  # keep legacy name for best-total
    ckpt_path_total = ckpt_path
    ckpt_path_ik = os.path.join(OUT_DIR, "best_ik.pt")
    ckpt_path_col = os.path.join(OUT_DIR, "best_col.pt")

    print("Start training…")
    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"[epoch {epoch}/{EPOCHS}] train", leave=False)
        loss_ik_sum = loss_col_sum = 0.0
        corr_ik = corr_col = 0
        seen = 0

        for batch in pbar:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            li, lc = model(
                batch["grasp_Oi"].float(),
                batch["objW_pick"].float(),
                batch["objW_place"].float(),
                meta=batch.get("meta", None).float() if USE_META and "meta" in batch else None,
                corners_f=batch.get("corners_f", None).float() if USE_CORNERS and "corners_f" in batch else None,
            )
            yik  = batch["y_ik"]
            ycol = batch["y_col"]

            loss_ik  = bce_with_class_weights(li, yik,  pos_w=IK_POS_W,  neg_w=IK_NEG_W)
            loss_col = bce_with_class_weights(lc, ycol, pos_w=COL_POS_W, neg_w=COL_NEG_W)
            loss = loss_ik + loss_col
            loss.backward()
            if CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

            # running stats
            with torch.no_grad():
                pred_ik  = (li.sigmoid()  >= 0.5).float()
                pred_col = (lc.sigmoid()  >= 0.5).float()
                corr_ik  += (pred_ik  == yik).sum().item()
                corr_col += (pred_col == ycol).sum().item()
                bsz = yik.shape[0]
                seen += bsz
                loss_ik_sum  += loss_ik.item()  * bsz
                loss_col_sum += loss_col.item() * bsz

            pbar.set_postfix({
                "L_ik": f"{loss_ik.item():.4f}",
                "L_col": f"{loss_col.item():.4f}",
                "acc_ik": f"{(corr_ik/seen):.3f}",
                "acc_col": f"{(corr_col/seen):.3f}",
                "lr": f"{sched.get_last_lr()[0]:.2e}",
            })

        sched.step()

        # epoch train summary
        tr_loss_ik  = loss_ik_sum / seen
        tr_loss_col = loss_col_sum / seen
        tr_acc_ik   = corr_ik / seen
        tr_acc_col  = corr_col / seen

        # val
        # Use UNWEIGHTED validation losses for selection/checkpointing
        valm = evaluate(model, dl_val, device, USE_META, USE_CORNERS,
                        ik_w=(IK_POS_W, IK_NEG_W), col_w=(COL_POS_W, COL_NEG_W), weighted=False)

        # print summary
        print(f"Epoch {epoch:02d} | "
              f"train L_ik {tr_loss_ik:.4f} acc_ik {tr_acc_ik:.3f} | "
              f"L_col {tr_loss_col:.4f} acc_col {tr_acc_col:.3f}  ||  "
              f"val L_ik {valm['loss_ik']:.4f} acc_ik {valm['acc_ik']:.3f} | "
              f"L_col {valm['loss_col']:.4f} acc_col {valm['acc_col']:.3f}")

        # log
        hist["tr_loss_ik"].append(tr_loss_ik)
        hist["tr_loss_col"].append(tr_loss_col)
        hist["tr_acc_ik"].append(tr_acc_ik)
        hist["tr_acc_col"].append(tr_acc_col)
        hist["va_loss_ik"].append(valm["loss_ik"])
        hist["va_loss_col"].append(valm["loss_col"])
        hist["va_acc_ik"].append(valm["acc_ik"])
        hist["va_acc_col"].append(valm["acc_col"])

        # checkpoint on total val loss
        val_total = valm["loss_ik"] + valm["loss_col"]
        if val_total < best_val_total:
            best_val_total = val_total
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "hist": hist,
                        "cfg": {
                            "USE_CORNERS": USE_CORNERS, "USE_META": USE_META, "USE_DELTA": USE_DELTA,
                            "HIDDEN": HIDDEN, "DROPOUT": DROPOUT
                        }},
                       ckpt_path_total)
            print(f"  ↳ saved best-total checkpoint: {ckpt_path_total}")
            no_improve = 0  # reset counter when total improves
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement {no_improve}/{PATIENCE}).")
                break

        # also save per-head bests (independent of early stopping)
        if valm["loss_ik"] < best_val_ik:
            best_val_ik = valm["loss_ik"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "hist": hist,
                        "cfg": {
                            "USE_CORNERS": USE_CORNERS, "USE_META": USE_META, "USE_DELTA": USE_DELTA,
                            "HIDDEN": HIDDEN, "DROPOUT": DROPOUT
                        }},
                       ckpt_path_ik)
            print(f"  ↳ saved best-ik checkpoint: {ckpt_path_ik}")

        if valm["loss_col"] < best_val_col:
            best_val_col = valm["loss_col"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "hist": hist,
                        "cfg": {
                            "USE_CORNERS": USE_CORNERS, "USE_META": USE_META, "USE_DELTA": USE_DELTA,
                            "HIDDEN": HIDDEN, "DROPOUT": DROPOUT
                        }},
                       ckpt_path_col)
            print(f"  ↳ saved best-col checkpoint: {ckpt_path_col}")

    # final test eval (best or last — we’ll use best)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    # Report unweighted test losses for comparability with selection
    testm = evaluate(model, dl_test, device, USE_META, USE_CORNERS,
                     ik_w=(IK_POS_W, IK_NEG_W), col_w=(COL_POS_W, COL_NEG_W), weighted=False)
    print(f"[TEST] L_ik {testm['loss_ik']:.4f} acc_ik {testm['acc_ik']:.3f} | "
          f"L_col {testm['loss_col']:.4f} acc_col {testm['acc_col']:.3f}")

    # plot curves
    fig = plt.figure(figsize=(10,6))
    xs = np.arange(1, len(hist["tr_loss_ik"]) + 1)
    # losses
    plt.subplot(2,2,1); plt.plot(xs, hist["tr_loss_ik"]);  plt.title("Train IK Loss")
    plt.subplot(2,2,2); plt.plot(xs, hist["tr_loss_col"]); plt.title("Train Collision Loss")
    plt.subplot(2,2,3); plt.plot(xs, hist["va_loss_ik"]);  plt.title("Val IK Loss")
    plt.subplot(2,2,4); plt.plot(xs, hist["va_loss_col"]); plt.title("Val Collision Loss")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "loss_curves.png")); plt.close(fig)

    fig = plt.figure(figsize=(10,6))
    plt.subplot(2,2,1); plt.plot(xs, hist["tr_acc_ik"]);  plt.title("Train IK Acc")
    plt.subplot(2,2,2); plt.plot(xs, hist["tr_acc_col"]); plt.title("Train Collision Acc")
    plt.subplot(2,2,3); plt.plot(xs, hist["va_acc_ik"]);  plt.title("Val IK Acc")
    plt.subplot(2,2,4); plt.plot(xs, hist["va_acc_col"]); plt.title("Val Collision Acc")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "acc_curves.png")); plt.close(fig)

    # log dump
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved plots & logs to: {OUT_DIR}")

    # Optional: evaluate SIM deck using the newly saved checkpoint
    if AUTO_EVAL_DECK:
        try:
            import importlib
            evalmod = importlib.import_module("placement_quality.path_simulation.model_training.evaluate")
            evalmod.USE_CORNERS = USE_CORNERS
            evalmod.USE_META = USE_META
            evalmod.USE_DELTA = USE_DELTA
            evalmod.USE_TRANSPORT = USE_TRANSPORT
            evalmod.DEFAULT_CHECKPOINT = ckpt_path
            # Ensure evaluation reads from the same precompute directory used for this run
            evalmod.PRECOMP_ROOT = PRECOMP_ROOT
            print("[Eval] Running deck evaluation with current toggles…")
            evalmod.main()
        except Exception as e:
            print(f"[Eval] Skipped deck evaluation due to error: {e}")

if __name__ == "__main__":
    main()
