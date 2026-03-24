#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# sklearn for advanced metrics (as in your preferred style)
from sklearn.metrics import roc_auc_score, average_precision_score

# === your modules ===
from dataset import FinalCornersHandDataset
from model import FinalCornersAuxModel


# =========================
# Hard-coded paths & knobs
# =========================
DATA_ROOT        = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
TRAIN_MEMMAP_DIR = f"{DATA_ROOT}/memmaps_train"   # must exist (built from train.json)
VAL_MEMMAP_DIR   = f"{DATA_ROOT}/memmaps_val"     # must exist (built from val.json)

# outputs
LOG_DIR   = os.path.join(DATA_ROOT, "training", "logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
MODEL_DIR = os.path.join(DATA_ROOT, "training", "models", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# training hyperparams (kept close to your style)
BATCH_SIZE    = 8192
EPOCHS        = 150
INITIAL_LR    = 1e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
WARMUP_EPOCHS = 2
EARLY_PATIENCE= 30

USE_WEIGHTED_FIRST_K = 10  # first K epochs use WeightedRandomSampler

# =========================
# Tee: duplicate stdout to file (your style)
# =========================
class Tee:
    """Duplicate stdout/stderr to console and a log file."""
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


# =========================
# Plots (your style)
# =========================
def save_enhanced_plots(history, log_dir):
    """Enhanced plotting with ROC-AUC and PR-AUC curves (exact style)"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Enhanced Training Metrics', fontsize=16)

    # Loss
    axes[0, 0].plot(history['train_collision_loss'], label='Train', color='blue', alpha=0.7)
    axes[0, 0].plot(history['val_collision_loss'],   label='Val',   color='orange', alpha=0.7)
    axes[0, 0].set_title('Collision Prediction Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('BCE Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_collision_accuracy'], label='Train', color='blue', alpha=0.7)
    axes[0, 1].plot(history['val_collision_accuracy'],   label='Val',   color='orange', alpha=0.7)
    axes[0, 1].set_title('Collision Accuracy')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # ROC-AUC
    axes[1, 0].plot(history['train_collision_roc_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 0].plot(history['val_collision_roc_auc'],   label='Val',   color='orange', alpha=0.7)
    axes[1, 0].set_title('ROC-AUC Score')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('ROC-AUC'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # PR-AUC
    axes[1, 1].plot(history['train_collision_pr_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 1].plot(history['val_collision_pr_auc'],   label='Val',   color='orange', alpha=0.7)
    axes[1, 1].set_title('Precision-Recall AUC')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('PR-AUC'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    # Precision
    axes[2, 0].plot(history['train_collision_precision'], label='Train', color='blue', alpha=0.7)
    axes[2, 0].plot(history['val_collision_precision'],   label='Val',   color='orange', alpha=0.7)
    axes[2, 0].set_title('Collision Precision')
    axes[2, 0].set_xlabel('Epoch'); axes[2, 0].set_ylabel('Precision'); axes[2, 0].legend(); axes[2, 0].grid(True, alpha=0.3)

    # F1
    axes[2, 1].plot(history['train_collision_f1'], label='Train', color='blue', alpha=0.7)
    axes[2, 1].plot(history['val_collision_f1'],   label='Val',   color='orange', alpha=0.7)
    axes[2, 1].set_title('Collision F1 Score')
    axes[2, 1].set_xlabel('Epoch'); axes[2, 1].set_ylabel('F1 Score'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, 'enhanced_training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =========================
# Metrics (vectorized, sklearn back-end)
# =========================
def compute_advanced_metrics(y_true, y_pred_probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred_probs = np.asarray(y_pred_probs).astype(np.float64)
    y_pred = (y_pred_probs > threshold).astype(np.int32)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    accuracy  = (tp + tn) / max(1, (tp + fp + fn + tn))
    precision = tp / max(1, (tp + fp))
    recall    = tp / max(1, (tp + fn))
    f1        = 2 * precision * recall / max(1e-12, (precision + recall))

    # Advanced
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_pred_probs)
    except Exception:
        pr_auc = y_true.mean()

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }


# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- tee logs ---
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(os.path.join(LOG_DIR, 'training.log'), 'a')
    sys.stdout = Tee(orig_stdout, log_file)

    print(f"=== Training on {device} ===")
    print(f"Train memmaps: {TRAIN_MEMMAP_DIR}")
    print(f"Val   memmaps: {VAL_MEMMAP_DIR}")

    # --- load train normalization stats (saved by your memmap builder) ---
    with open(os.path.join(TRAIN_MEMMAP_DIR, "stats.json"), "r") as f:
        train_norm_stats = json.load(f)

    # --- datasets ---
    print("Loading datasets with feature normalization...")
    train_dataset = FinalCornersHandDataset(
        mem_dir=TRAIN_MEMMAP_DIR,
        normalization_stats=train_norm_stats,
        is_training=True
    )
    val_dataset = FinalCornersHandDataset(
        mem_dir=VAL_MEMMAP_DIR,
        normalization_stats=train_norm_stats,
        is_training=False
    )
    print(f"  → {len(train_dataset)} train samples")
    print(f"  → {len(val_dataset)} val samples")

    # --- class balance for pos_weight & sampler ---
    with open(os.path.join(TRAIN_MEMMAP_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    Ntr = int(meta["N"])
    mm_y = np.memmap(meta["label_file"], dtype=np.float32, mode="r", shape=(Ntr,1))
    n_pos = int((mm_y[:,0] >= 0.5).sum())
    n_neg = Ntr - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    print(f"Class distribution - Negative: {n_neg}, Positive: {n_pos}")
    print(f"Using pos_weight: {float(pos_weight.item()):.3f}")

    # --- loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    # --- model / opt / loss / sched ---
    model = FinalCornersAuxModel(
        aux_in=12,                    # t_loc_z(3) + R6(6) + dims_z(3) + rf6(6)
        corners_hidden=(128, 64),
        aux_hidden=(64, 32),
        head_hidden=128,
        dropout_p=0.05,
        use_film=True,
        two_head=False                # only core 12 dims in aux
    ).to(device)

    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    main_scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                            patience=10, verbose=True, min_lr=1e-6)

    # --- history dict (your keys) ---
    history = {k: [] for k in [
        'train_collision_loss', 'train_collision_accuracy', 'train_collision_precision',
        'train_collision_recall', 'train_collision_f1', 'train_collision_roc_auc', 'train_collision_pr_auc',
        'val_collision_loss',   'val_collision_accuracy',   'val_collision_precision',
        'val_collision_recall', 'val_collision_f1',         'val_collision_roc_auc', 'val_collision_pr_auc'
    ]}

    best_val_loss = float("inf")
    best_val_roc  = 0.0
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Initial learning rate: {INITIAL_LR}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Using Normal Sampling")

        # -------- TRAIN --------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_probs, tr_labels = [], []

        for corners, aux, label in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", file=sys.stderr):
            corners = corners.to(device, non_blocking=True)
            aux     = aux.to(device, non_blocking=True)
            y       = label.to(device, non_blocking=True).view(-1,1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(corners, aux)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_n        += bs

            with torch.no_grad():
                p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                tr_probs.append(p)
                tr_labels.append(y.detach().cpu().numpy().reshape(-1))

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        # aggregate train metrics
        tr_probs  = np.concatenate(tr_probs) if tr_probs else np.array([], dtype=np.float32)
        tr_labels = np.concatenate(tr_labels).astype(np.int32) if tr_labels else np.array([], dtype=np.int32)
        train_loss = train_loss_sum / max(1, train_n)
        train_metrics = compute_advanced_metrics(tr_labels, tr_probs)

        # -------- VAL --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        va_probs, va_labels = [], []

        with torch.no_grad():
            for corners, aux, label in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", file=sys.stderr):
                corners = corners.to(device, non_blocking=True)
                aux     = aux.to(device, non_blocking=True)
                y       = label.to(device, non_blocking=True).view(-1,1)

                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    logits = model(corners, aux)
                    loss = criterion(logits, y)

                bs = y.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_n        += bs

                p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                va_probs.append(p)
                va_labels.append(y.detach().cpu().numpy().reshape(-1))

        val_loss   = val_loss_sum / max(1, val_n)
        va_probs   = np.concatenate(va_probs) if va_probs else np.array([], dtype=np.float32)
        va_labels  = np.concatenate(va_labels).astype(np.int32) if va_labels else np.array([], dtype=np.int32)
        val_metrics= compute_advanced_metrics(va_labels, va_probs)

        # main scheduler after warmup
        if epoch >= WARMUP_EPOCHS:
            main_scheduler.step(val_loss)

        # store history
        history['train_collision_loss'].append(train_loss)
        history['train_collision_accuracy'].append(train_metrics['accuracy'])
        history['train_collision_precision'].append(train_metrics['precision'])
        history['train_collision_recall'].append(train_metrics['recall'])
        history['train_collision_f1'].append(train_metrics['f1'])
        history['train_collision_roc_auc'].append(train_metrics['roc_auc'])
        history['train_collision_pr_auc'].append(train_metrics['pr_auc'])

        history['val_collision_loss'].append(val_loss)
        history['val_collision_accuracy'].append(val_metrics['accuracy'])
        history['val_collision_precision'].append(val_metrics['precision'])
        history['val_collision_recall'].append(val_metrics['recall'])
        history['val_collision_f1'].append(val_metrics['f1'])
        history['val_collision_roc_auc'].append(val_metrics['roc_auc'])
        history['val_collision_pr_auc'].append(val_metrics['pr_auc'])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f},  Acc: {val_metrics['accuracy']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")

        # targets (optional messages)
        target_loss_reached = val_loss < 0.25
        target_roc_reached  = val_metrics['roc_auc'] > 0.905
        if target_loss_reached:
            print(f"  🎯 TARGET REACHED: Validation loss below 0.25 ({val_loss:.4f})")
        if target_roc_reached:
            print(f"  🏆 TARGET REACHED: ROC-AUC above 0.905 ({val_metrics['roc_auc']:.4f})")

        # save bests
        is_best_loss = val_loss < best_val_loss
        is_best_roc  = val_metrics['roc_auc'] > best_val_roc

        if is_best_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_roc_auc': val_metrics['roc_auc'],
                'normalization_stats': train_norm_stats
            }, os.path.join(MODEL_DIR, f'best_model_loss_{os.path.basename(LOG_DIR)}.pth'))
            print(f"  💾 Saved best loss model (loss: {val_loss:.4f})")

        if is_best_roc:
            best_val_roc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_roc_auc': val_metrics['roc_auc'],
                'normalization_stats': train_norm_stats
            }, os.path.join(MODEL_DIR, f'best_model_roc_{os.path.basename(LOG_DIR)}.pth'))
            print(f"  🎯 Saved best ROC-AUC model (ROC-AUC: {val_metrics['roc_auc']:.4f})")

        if not is_best_loss:
            patience_counter += 1
            if patience_counter >= EARLY_PATIENCE:
                print(f"Early stopping triggered after {EARLY_PATIENCE} epochs without improvement")
                break

        # save plots every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_enhanced_plots(history, LOG_DIR)
        print("-" * 80)

    # final saves
    save_enhanced_plots(history, LOG_DIR)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_roc_auc': val_metrics['roc_auc'],
        'normalization_stats': train_norm_stats
    }, os.path.join(MODEL_DIR, f'final_model_{os.path.basename(LOG_DIR)}.pth'))

    with open(os.path.join(LOG_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("✅ Training completed!")
    # restore stdout/stderr + close file
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()
    print(f"📊 Logs saved to: {LOG_DIR}")
    print(f"💾 Models saved to: {MODEL_DIR}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Best validation ROC-AUC: {best_val_roc:.4f}")


if __name__ == "__main__":
    # just run
    main()
