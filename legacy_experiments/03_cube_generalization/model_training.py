#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train FinalCornersAuxModel on memmapped v6 data:
  X = [ corners_24 (z-scored), aux_12 = t_loc_z(3) + R_loc6(6 raw) + dims_z(3) ]
  y = collision (1 if collision, else 0)

Hard-coded paths & params. No CLI required.
"""

import os, json, math, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
# --- your modules (expecting the memmap dataset + model you already have) ---
from dataset import FinalCornersHandDataset     # expects mem_dir + stats
from model import FinalCornersAuxModel          # K=12, FiLM on, single head


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT         = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
TRAIN_MEMMAP_DIR  = f"{DATA_ROOT}/memmaps_train"   # build these once with your dataset builder
VAL_MEMMAP_DIR    = f"{DATA_ROOT}/memmaps_val"     # (train stats must be saved in TRAIN_MEMMAP_DIR/stats.json)
OUTPUT_ROOT       = f"{DATA_ROOT}/training"

AUX_OPTION        = 1    # 0=base, 1=+R_of6, 2=+tloc_f_inplane(2), 3=+R_hf6

# Per-run directory (configurable). Change RUN_NAME or set env RUN_NAME.
RUN_NAME          = "R_of6"
RUN_DIR           = os.path.join(OUTPUT_ROOT, RUN_NAME)

SEED              = 42
BATCH_SIZE        = 8192
EPOCHS            = 150
LR                = 5e-4
WEIGHT_DECAY      = 1e-4
PATIENCE          = 20             # early stop on ROC-AUC
NUM_WORKERS       = 16
GRAD_CLIP         = 1.0
USE_AMP           = True


# =========================
# Utilities
# =========================
class Tee:
    """Duplicate stdout to console and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False

def save_enhanced_plots(history, log_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Enhanced Training Metrics', fontsize=16)

    axes[0, 0].plot(history['train_collision_loss'], label='Train', color='blue', alpha=0.7)
    axes[0, 0].plot(history['val_collision_loss'],   label='Val',   color='orange', alpha=0.7)
    axes[0, 0].set_title('Collision Prediction Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('BCE Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_collision_accuracy'], label='Train', color='blue', alpha=0.7)
    axes[0, 1].plot(history['val_collision_accuracy'],   label='Val',   color='orange', alpha=0.7)
    axes[0, 1].set_title('Collision Accuracy')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_collision_roc_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 0].plot(history['val_collision_roc_auc'],   label='Val',   color='orange', alpha=0.7)
    axes[1, 0].set_title('ROC-AUC Score')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('ROC-AUC'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['train_collision_pr_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 1].plot(history['val_collision_pr_auc'],   label='Val',   color='orange', alpha=0.7)
    axes[1, 1].set_title('Precision-Recall AUC')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('PR-AUC'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # precision and F1 not tracked here; mirror layout with blanks
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')

    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    out_png = os.path.join(log_dir, 'enhanced_training_metrics.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dirs():
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)


# ---- metrics (NumPy; no sklearn dependency) ----
def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y = y_true[order].astype(np.int32)
    P = int(y.sum())
    N = y.size - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / (P + 1e-12)
    fpr = fp / (N + 1e-12)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = _roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y = y_true[order].astype(np.int32)
    P = int(y.sum())
    if P == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (P + 1e-12)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p, r = _precision_recall_curve(y_true, y_score)
    return float(np.trapz(p, r))


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid=200):
    """Scan thresholds to maximize balanced accuracy and F1 on val."""
    best_bal = (-1.0, 0.5)  # (score, tau)
    best_f1  = (-1.0, 0.5)
    for tau in np.linspace(0.0, 1.0, grid):
        y_hat = (y_prob >= tau).astype(np.int32)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        bal = 0.5 * (tpr + tnr)
        prec = tp / (tp + fp + 1e-12)
        rec  = tpr
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        if bal > best_bal[0]: best_bal = (bal, float(tau))
        if f1  > best_f1[0]:  best_f1  = (f1,  float(tau))
    return {"bal": best_bal, "f1": best_f1}


# =========================
# Epoch loops
# =========================
def run_epoch(model, loader, optimizer, scaler, criterion, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_probs, all_labels = [], []

    it = tqdm(loader, desc="[Train]" if train else "[Val]", leave=False)
    for corners, aux, y in it:
        corners = corners.to(device, non_blocking=True)      # [B,24]
        aux     = aux.to(device, non_blocking=True)          # [B,12]
        label   = y.to(device, non_blocking=True).view(-1,1) # [B,1]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
            logits = model(corners, aux)                     # [B,1]
            loss = criterion(logits, label)

        if train:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits).view(-1)           # [B]
            preds = (probs >= 0.5).float().view(-1,1)
            total_correct += int((preds.eq(label)).sum().item())
            total_count   += label.size(0)
            total_loss    += float(loss.item()) * label.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(label.view(-1).detach().cpu().numpy())

    if total_count == 0:
        return {"loss":0.0, "acc":0.0, "roc_auc":0.0, "pr_auc":0.0, "n":0}, np.array([]), np.array([])

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(np.int32)
    stats = {
        "loss":   total_loss / total_count,
        "acc":    total_correct / total_count,
        "roc_auc": roc_auc(labels, probs),
        "pr_auc":  pr_auc(labels, probs),
        "n":      total_count,
    }
    return stats, probs, labels


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging setup (Tee)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RUN_DIR, "training.log")
    log_file = open(log_path, "a")
    # additional, separate logs (minimal change): metrics and thresholds
    metrics_log_path = os.path.join(RUN_DIR, "metrics_log.txt")
    thresholds_log_path = os.path.join(RUN_DIR, "threshold_log.txt")
    metrics_file = open(metrics_log_path, "a")
    thresholds_file = open(thresholds_log_path, "a")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_file)

    # ---- load train stats (train-only!) ----
    stats_json = os.path.join(TRAIN_MEMMAP_DIR, "stats.json")
    with open(stats_json, "r") as f:
        train_stats = json.load(f)

    # ---- datasets ----
    print("Loading datasets (memmaps) with train normalization...")
    train_ds = FinalCornersHandDataset(
        mem_dir=TRAIN_MEMMAP_DIR,
        normalization_stats=train_stats,
        is_training=True,
        feature_option=AUX_OPTION + 1
    )
    val_ds   = FinalCornersHandDataset(
        mem_dir=VAL_MEMMAP_DIR,
        normalization_stats=train_stats,
        is_training=False,
        feature_option=AUX_OPTION + 1
    )
    print(f"  → {len(train_ds)} train samples")
    print(f"  → {len(val_ds)} val samples")

    # ---- loaders ----
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=False, persistent_workers=True
    )

    # ---- class balance (pos = collision) for pos_weight ----
    # read from train memmap directly
    meta = json.load(open(os.path.join(TRAIN_MEMMAP_DIR, "meta.json"), "r"))
    Ntr = int(meta["N"])
    mm_y = np.memmap(meta["label_file"], dtype=np.float32, mode="r", shape=(Ntr,1))
    n_pos = int((mm_y[:,0] >= 0.5).sum())
    n_neg = Ntr - n_pos
    pos_w = float(n_neg / max(1, n_pos))
    print(f"Class balance (train): pos={n_pos}  neg={n_neg}  → pos_weight={pos_w:.3f}")

    # map option → aux dims
    _aux_in = 12
    if AUX_OPTION == 1:   # +R_of6
        _aux_in = 18
    elif AUX_OPTION == 2: # +tloc_f_inplane(2)
        _aux_in = 14
    elif AUX_OPTION == 3: # +R_hf6
        _aux_in = 18

    # ---- model / opt / loss / sched ----
    model = FinalCornersAuxModel(
        aux_in=_aux_in,
        corners_hidden=(128,64),
        aux_hidden=(64,32),
        head_hidden=128,
        dropout_p=0.05,
        use_film=True,
        two_head=False
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    scaler = GradScaler(enabled=USE_AMP and torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.1)

    # ---- logging / checkpointing ----
    ckpt_path= os.path.join(RUN_DIR, "best_roc.pt")
    best_auc = -math.inf
    best_epoch = -1
    best_loss = math.inf
    best_loss_path = os.path.join(RUN_DIR, "best_loss.pt")
    patience_counter = 0
    # history for plotting
    history = {k: [] for k in [
        'train_collision_loss', 'train_collision_accuracy', 'train_collision_roc_auc', 'train_collision_pr_auc',
        'val_collision_loss',   'val_collision_accuracy',   'val_collision_roc_auc',   'val_collision_pr_auc']}

    head = (f"run={run_id}  device={device}\n"
            f"data_root={DATA_ROOT}\n"
            f"train_memmaps={TRAIN_MEMMAP_DIR}\nval_memmaps={VAL_MEMMAP_DIR}\n"
            f"batch={BATCH_SIZE} epochs={EPOCHS} lr={LR} wd={WEIGHT_DECAY}\n")
    print(head)

    for epoch in range(EPOCHS):
        # train
        tr_stats, _, _ = run_epoch(model, train_loader, optimizer, scaler, criterion, device, train=True)
        # val
        with torch.no_grad():
            va_stats, va_probs, va_labels = run_epoch(model, val_loader, optimizer, scaler, criterion, device, train=False)

        scheduler.step()

        # log message
        msg = (f"Epoch {epoch+1:03d}/{EPOCHS} | "
               f"Train: loss={tr_stats['loss']:.4f} acc={tr_stats['acc']:.4f} "
               f"| Val: loss={va_stats['loss']:.4f} acc={va_stats['acc']:.4f} "
               f"ROC-AUC={va_stats['roc_auc']:.4f} PR-AUC={va_stats['pr_auc']:.4f} "
               f"| LR={scheduler.get_last_lr()[0]:.2e}")
        print(msg)
        # write per-epoch metrics to separate metrics log
        metrics_file.write(msg + "\n")
        metrics_file.flush()

        # record history for plots
        history['train_collision_loss'].append(tr_stats['loss'])
        history['train_collision_accuracy'].append(tr_stats['acc'])
        history['train_collision_roc_auc'].append(tr_stats['roc_auc'])
        history['train_collision_pr_auc'].append(tr_stats['pr_auc'])
        history['val_collision_loss'].append(va_stats['loss'])
        history['val_collision_accuracy'].append(va_stats['acc'])
        history['val_collision_roc_auc'].append(va_stats['roc_auc'])
        history['val_collision_pr_auc'].append(va_stats['pr_auc'])

        # save interim plots every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_enhanced_plots(history, RUN_DIR)

        # best on ROC-AUC
        cur_auc = va_stats["roc_auc"]
        if cur_auc > best_auc + 1e-5:
            best_auc = cur_auc
            best_epoch = epoch
            patience_counter = 0

            # choose thresholds on val and log confusion tables
            thr = pick_threshold(va_labels, va_probs, grid=301)
            tau_bal = thr["bal"][1]; tau_f1 = thr["f1"][1]
            def _conf(y, p, t):
                yhat = (p >= t).astype(np.int32)
                tp = int(((yhat==1)&(y==1)).sum())
                tn = int(((yhat==0)&(y==0)).sum())
                fp = int(((yhat==1)&(y==0)).sum())
                fn = int(((yhat==0)&(y==1)).sum())
                acc = (tp+tn)/max(1,(tp+tn+fp+fn))
                prec = tp/max(1,(tp+fp)); rec = tp/max(1,(tp+fn))
                f1 = 2*prec*rec/max(1e-12,(prec+rec))
                return dict(tp=tp, fp=fp, tn=tn, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)
            conf_bal = _conf(va_labels, va_probs, tau_bal)
            conf_f1  = _conf(va_labels, va_probs, tau_f1)

            # save checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "normalization_stats": train_stats,     # train-only stats
                "val_roc_auc": float(best_auc),
                "val_loss": float(va_stats["loss"]),
                "tau_bal": float(tau_bal),
                "tau_f1":  float(tau_f1),
                "conf_bal": conf_bal,
                "conf_f1":  conf_f1,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED
                },
                "model_type": "FinalCornersAuxModel(K=12, FiLM, single-head)",
            }
            torch.save(ckpt, ckpt_path)

            print(f"[save] best ckpt → {ckpt_path}")
            print(f"[val] tau_bal={tau_bal:.3f} conf={conf_bal}")
            print(f"[val] tau_f1 ={tau_f1:.3f} conf={conf_f1}")
            # write thresholds and their confusion stats to separate threshold log
            thresholds_file.write(
                f"Epoch {epoch+1:03d} | ROC-AUC={cur_auc:.4f}\n"
            )
            thresholds_file.write(
                f"[val] tau_bal={tau_bal:.3f} conf={conf_bal}\n"
            )
            thresholds_file.write(
                f"[val] tau_f1 ={tau_f1:.3f} conf={conf_f1}\n"
            )
            thresholds_file.flush()
        # best on Val Loss
        if va_stats["loss"] < best_loss - 1e-8:
            best_loss = va_stats["loss"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "normalization_stats": train_stats,
                "val_roc_auc": float(cur_auc),
                "val_loss": float(best_loss)
            }, best_loss_path)
            print(f"[save] best-loss ckpt → {best_loss_path} (loss={best_loss:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best @ {best_epoch+1} with ROC-AUC={best_auc:.4f})")
                break

    # final plot
    save_enhanced_plots(history, RUN_DIR)
    print(f"✅ Training complete. Best ROC-AUC={best_auc:.4f} @ epoch {best_epoch+1}")
    print(f"📝 Log:   {log_path}")
    # also save final model under run models dir
    final_ckpt_path = os.path.join(RUN_DIR, "final.pt")
    torch.save({
        "epoch": best_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalization_stats": train_stats,
        "best_val_roc_auc": float(best_auc)
    }, final_ckpt_path)
    print(f"💾 Best-ROC checkpoint: {ckpt_path}")
    print(f"💾 Best-Loss checkpoint: {best_loss_path}")
    print(f"💾 Final checkpoint: {final_ckpt_path}")
    # restore stdout and close log
    sys.stdout = orig_stdout
    log_file.close()
    metrics_file.close()
    thresholds_file.close()


if __name__ == "__main__":
    main()
