#!/usr/bin/env python3
"""
ENIGMA-style EEG Brain Decoding Classification Model — v1 (TinyENIGMA, no SGCN)
================================================================================
This is the train_enigma.py depthwise-CNN model (EEGNet-inspired "TinyENIGMA"),
modified to include the v2 additions:
  - per-subject per-channel re-weighting from `eeg_channel_weights.csv`
  - subjects 5 / 8 / 12 excluded (faulty per the dataset notes)
  - full plotting suite (training curves, confusion matrices, per-class bars,
    weight heatmap, CV summary)

It keeps train_enigma.py's lazy disk-backed caching: each (C, T) window lives on
disk as a contiguous float32 array and exactly one window is read per __getitem__.

Bug fixes carried over / applied:
  - `best_state` starts from best_acc=-1.0 so epoch 1 always snapshots
    (avoids load_state_dict(None) when val-acc never beats 0).
  - subject-id parsing is zero-padded so weight lookups don't silently miss.
  - windows are copied before in-place augmentation.
"""

import argparse
import hashlib
import re
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", message="Physical range is not defined")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    sys.exit("pip install mne")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit("pip install torch")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    sys.exit("pip install scikit-learn")

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("  [PLOT] matplotlib not found — plots disabled. pip install matplotlib")

import functools
print = functools.partial(print, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────────────────────────────────────
N_CLASSES          = 36
N_INSTANCES        = 5
DROPOUT_P          = 0.5
EXCLUDED_SUBJECTS  = {5, 8, 12}
ALL_SUBJECTS       = set(range(1, 13))

# ──────────────────────────────────────────────────────────────────────────────
# fNIRS-configuration subject groups (excluded subs 5/8/12 are in neither group)
# ──────────────────────────────────────────────────────────────────────────────
GROUP_A_SUBJECTS = {1, 2, 3, 4, 6}      # fNIRS configuration A
GROUP_B_SUBJECTS = {7, 9, 10, 11}       # fNIRS configuration B
GROUPS = [
    ("groupA_1-2-3-4-6", GROUP_A_SUBJECTS),
    ("groupB_7-9-10-11", GROUP_B_SUBJECTS),
]

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

RESAMPLE_FREQ      = 256.0
BANDPASS_LOW       = 1.0
BANDPASS_HIGH      = 40.0
NOTCH_FREQ         = 50.0
ARTIFACT_THRESH_UV = 150e-6
CACHE_DIR          = Path(".eeg_cache_v8")     # shared, contiguous-float32 format
PLOT_DIR           = Path("enigma_plots_v1_split")

MODEL_TAG   = "v1-split (TinyENIGMA, per-fNIRS-group models)"
CKPT_PREFIX = "enigma_v1_split"


# ──────────────────────────────────────────────────────────────────────────────
# eeg_channel_weights
# ──────────────────────────────────────────────────────────────────────────────

def load_channel_weights(csv_path: Path, allowed_subjects: set) -> dict:
    df = pd.read_csv(str(csv_path), index_col=0)
    weights = {}
    for col in df.columns:
        match = re.search(r"(\d+)", str(col))
        if not match:
            continue
        sub_num = int(match.group(1))
        if sub_num not in allowed_subjects:
            continue
        sub_key  = f"sub-{sub_num:02d}"
        col_vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        if np.all(np.isnan(col_vals)):
            continue
        col_vals = np.where(np.isnan(col_vals), 1.0, col_vals).astype(np.float32)
        weights[sub_key] = col_vals
    tqdm.write(f"  [WEIGHTS] Loaded channel weights for {len(weights)} subjects: "
               f"{sorted(weights.keys())}")
    return weights


def get_subject_id(bdf_path: Path) -> str:
    for part in bdf_path.parts:
        m = re.match(r"^sub-?(\d+)$", part)
        if m:
            return f"sub-{int(m.group(1)):02d}"
    return "unknown"


def apply_channel_weights(window: np.ndarray, sub_id: str,
                          weight_table: dict) -> np.ndarray:
    if sub_id not in weight_table:
        return window
    w = weight_table[sub_id]
    if w.shape[0] != window.shape[0]:
        tqdm.write(f"  [WEIGHTS] shape mismatch for {sub_id} "
                   f"({w.shape[0]} vs {window.shape[0]}) — skipping")
        return window
    return (window * w[:, np.newaxis]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (lazy: returns the Path to a contiguous float32 cache file)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_bdf(path: Path, window_sec: float, overlap_sec: float,
                   use_cache: bool = True) -> Path:
    key = hashlib.md5(
        f"{path}{window_sec}{overlap_sec}{RESAMPLE_FREQ}"
        f"{BANDPASS_LOW}{BANDPASS_HIGH}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{key}.npy"

    if use_cache and cache_path.exists():
        # Validate that it is a readable, non-object array; else reprocess.
        try:
            _ = np.load(str(cache_path), mmap_mode="r", allow_pickle=False)
            return cache_path
        except Exception:
            pass

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose=False)

    drop = [ch for ch in raw.ch_names
            if ch.upper().startswith(("EXG", "GSR", "STATUS", "TRIG",
                                       "AIO", "ERG", "RESP", "PLET", "TEMP"))]
    if drop:
        raw.drop_channels(drop)
    raw.pick_types(eeg=True, exclude="bads")

    try:
        raw.set_montage(mne.channels.make_standard_montage("biosemi64"),
                        on_missing="ignore", verbose=False)
    except Exception:
        pass

    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, method="fir", verbose=False)
    try:
        raw.notch_filter(NOTCH_FREQ, verbose=False)
    except Exception:
        pass

    if raw.info["sfreq"] > RESAMPLE_FREQ:
        raw.resample(RESAMPLE_FREQ, npad="auto")

    raw.set_eeg_reference("average", projection=False, verbose=False)

    data     = raw.get_data().astype(np.float32)
    target_T = int(RESAMPLE_FREQ * window_sec)
    step_T   = max(1, int(RESAMPLE_FREQ * (window_sec - overlap_sec)))

    valid_windows = []
    for start in range(0, data.shape[1] - target_T + 1, step_T):
        w = data[:, start: start + target_T].copy()
        if np.ptp(w, axis=1).max() > ARTIFACT_THRESH_UV:
            continue
        bs = max(1, int(target_T * 0.2))
        w -= w[:, :bs].mean(axis=1, keepdims=True)
        w  = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
        valid_windows.append(w.astype(np.float32))

    if use_cache and valid_windows:
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(str(cache_path), np.stack(valid_windows).astype(np.float32))

    return cache_path


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (lazy, disk-backed)
# ──────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, samples: list, window_sec: float, overlap_sec: float,
                 weight_table: dict, augment_data: bool = False,
                 use_cache: bool = True):
        self.augment_data = augment_data
        self.weight_table = weight_table
        self.items        = []   # (cache_path, w_idx, label, sub_id)
        rejected          = 0

        for path, label in samples:
            sub_id     = get_subject_id(path)
            ov         = overlap_sec if augment_data else 0.0
            cache_path = preprocess_bdf(path, window_sec, ov, use_cache)
            if not cache_path.exists():
                rejected += 1
                continue
            try:
                arr = np.load(str(cache_path), mmap_mode="r", allow_pickle=False)
                for w_idx in range(arr.shape[0]):
                    self.items.append((cache_path, w_idx, label, sub_id))
            except Exception:
                rejected += 1

        if rejected:
            tqdm.write(f"    Artifact rejection: {rejected} files removed/failed")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cache_path, w_idx, label, sub_id = self.items[idx]
        arr  = np.load(str(cache_path), mmap_mode="c", allow_pickle=False)
        data = np.array(arr[w_idx], dtype=np.float32, copy=True)   # independent copy
        data = apply_channel_weights(data, sub_id, self.weight_table)

        x = torch.from_numpy(data).unsqueeze(0)   # (1, C, T)
        if self.augment_data:
            x = x + torch.randn_like(x) * 0.05
            mask = (torch.rand(x.shape[1], 1) > 0.10).float()
            x   = x * mask.unsqueeze(0)
            x   = x * (0.9 + 0.2 * torch.rand(1).item())
        return x, label


# ──────────────────────────────────────────────────────────────────────────────
# Fold builder
# ──────────────────────────────────────────────────────────────────────────────

def build_fold(stimuli_dir: Path, test_instance: int,
               window_sec: float, overlap_sec: float,
               weight_table: dict, allowed_subjects: set, use_cache: bool):
    label_map           = {}
    train_raw, test_raw = [], []

    classes = sorted([d.name for d in stimuli_dir.iterdir() if d.is_dir()])
    for cls in classes:
        lbl            = len(label_map)
        label_map[cls] = lbl
        cls_dir        = stimuli_dir / cls

        for sub_dir in sorted(cls_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            m = re.match(r"sub-?(\d+)", sub_dir.name)
            if not m or int(m.group(1)) not in allowed_subjects:
                continue
            for inst in range(1, N_INSTANCES + 1):
                bdf = sub_dir / f"instance{inst}.bdf"
                if not bdf.exists():
                    continue
                if inst == test_instance:
                    test_raw.append((bdf, lbl))
                else:
                    train_raw.append((bdf, lbl))

    tqdm.write(f"    Building train set ({len(train_raw)} BDFs) ...")
    train_ds = EEGDataset(train_raw, window_sec, overlap_sec,
                          weight_table, augment_data=True,  use_cache=use_cache)
    tqdm.write(f"    Building test  set ({len(test_raw)} BDFs) ...")
    test_ds  = EEGDataset(test_raw,  window_sec, 0.0,
                          weight_table, augment_data=False, use_cache=use_cache)
    tqdm.write(f"    Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")
    return train_ds, test_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# Model — TinyENIGMA (EEGNet-inspired depthwise backbone)
# ──────────────────────────────────────────────────────────────────────────────

class TinyENIGMA(nn.Module):
    def __init__(self, n_channels, n_times, n_classes,
                 n_f=16, e_dim=8, dropout_p=DROPOUT_P, embed_dim=128):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_f, kernel_size=(1, 32),
                                       padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(n_f)
        self.spatial_conv = nn.Conv2d(n_f, n_f * 2, kernel_size=(n_channels, 1),
                                      groups=n_f, bias=False)
        self.bn2 = nn.BatchNorm2d(n_f * 2)
        self.pool    = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(dropout_p)
        self.proj    = nn.Conv2d(n_f * 2, e_dim, kernel_size=(1, 1), bias=False)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat  = self._forward_features(dummy).shape[1]

        self.embed_head = nn.Sequential(
            nn.LayerNorm(flat),
            nn.Linear(flat, embed_dim),
            nn.ELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim, n_classes),
        )

    def _forward_features(self, x):
        z = F.elu(self.bn1(self.temporal_conv(x)))
        z = self.dropout(F.elu(self.bn2(self.spatial_conv(z))))
        z = self.pool(self.proj(z))
        return z.flatten(1)

    def encode(self, x):
        return self.embed_head(self._forward_features(x))

    def forward(self, x):
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        B, device = features.shape[0], features.device
        sim       = torch.mm(features, features.T) / self.temp
        self_mask = torch.eye(B, device=device).bool()
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        sim       = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos     = pos_mask.float().sum(dim=1)
        valid     = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = -(log_prob * pos_mask.float()).sum(dim=1)
        return (loss[valid] / n_pos[valid]).mean()


# ──────────────────────────────────────────────────────────────────────────────
# One fold
# ──────────────────────────────────────────────────────────────────────────────

def run_fold(fold: int, stimuli_dir: Path, weight_table: dict,
             allowed_subjects: set, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec,
        weight_table, allowed_subjects, not args.no_cache)

    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data, skipping.")
        return None, None, None, label_map, None, None, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    n_channels = train_ds[0][0].shape[1]
    target_T   = train_ds[0][0].shape[-1]

    model  = TinyENIGMA(n_channels, target_T, N_CLASSES).to(DEVICE)
    supcon = SupConLoss()
    ce     = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: SupCon pretraining ───────────────────────────────────────────
    opt_pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, T_max=args.pretrain_epochs)

    pretrain_losses = []
    pre_bar = tqdm(range(1, args.pretrain_epochs + 1),
                   desc=f"  Fold {fold} pretrain", ncols=90,
                   unit="ep", leave=False, colour="yellow")
    for _ in pre_bar:
        model.train()
        ep_loss, batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_pre.zero_grad()
            feat = F.normalize(model.encode(xb), dim=1)
            loss = supcon(feat, yb)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_pre.step()
                ep_loss += loss.item(); batches += 1
        sch_pre.step()
        if batches:
            avg = ep_loss / batches
            pretrain_losses.append(avg)
            pre_bar.set_postfix(loss=f"{avg:.3f}")

    # ── Phase 2: classifier fine-tune, then unfreeze ──────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    opt_ft = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01)
    unfreeze_epoch       = args.epochs // 2
    best_acc, best_state = -1.0, None
    train_accs, val_accs = [], []

    ft_bar = tqdm(range(1, args.epochs + 1),
                  desc=f"  Fold {fold} finetune", ncols=90,
                  unit="ep", leave=False, colour="cyan")
    for epoch in ft_bar:
        if epoch == unfreeze_epoch:
            for p in model.parameters():
                p.requires_grad_(True)
            opt_ft = torch.optim.AdamW(model.parameters(),
                                       lr=args.lr * 0.1, weight_decay=0.01)

        model.train()
        tr_correct, tr_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_ft.zero_grad()
            logits = model(xb)
            ce(logits, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total   += xb.size(0)

        tr_acc  = tr_correct / tr_total  if tr_total  > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        train_accs.append(tr_acc); val_accs.append(val_acc)
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ft_bar.set_postfix(tr=f"{tr_acc:.2f}", val=f"{val_acc:.2f}",
                           best=f"{max(best_acc,0):.2f}")

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_labels.extend(yb.tolist())

    acc    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    curves = dict(pretrain_loss=pretrain_losses, train_acc=train_accs, val_acc=val_accs)
    return acc, all_preds, all_labels, label_map, best_state, n_channels, curves


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(fold, curves, best_acc, plot_dir):
    if not MATPLOTLIB_OK: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Fold {fold} — Training Curves (best val acc = {best_acc:.3f})",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(curves["pretrain_loss"], color="#e67e22", linewidth=1.5)
    ax.set_title("SupCon Pretraining Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    ax = axes[1]
    ax.plot(curves["train_acc"], label="Train", color="#2980b9", linewidth=1.5)
    ax.plot(curves["val_acc"],   label="Val",   color="#27ae60", linewidth=1.5)
    ax.axhline(1 / N_CLASSES, ls="--", color="gray", linewidth=1, label="Chance")
    ax.set_title("Fine-tune Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    out = plot_dir / f"fold{fold}_training_curves.png"
    fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def plot_per_class_accuracy(rows, plot_dir, title="Per-Class Accuracy"):
    if not MATPLOTLIB_OK or not rows: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in rows]; accs = [r[3] for r in rows]
    colors = ["#27ae60" if a >= 0.10 else "#e74c3c" for a in accs]
    fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.35)))
    bars = ax.barh(labels, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(1 / N_CLASSES, ls="--", color="black", linewidth=1.2, label="Chance")
    ax.set_xlim(0, max(accs) * 1.15 if accs else 1)
    ax.set_xlabel("Top-1 Accuracy"); ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    out = plot_dir / "per_class_accuracy.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def plot_confusion_matrix(all_labels, all_preds, label_map, fold, plot_dir):
    if not MATPLOTLIB_OK: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx_to_label = {v: k for k, v in label_map.items()}
    names        = [idx_to_label[i] for i in range(N_CLASSES)]
    cm           = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASSES)))
    cm_norm      = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold} — Normalised Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    out = plot_dir / f"fold{fold}_confusion_matrix.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def plot_channel_weight_heatmap(weight_table, plot_dir):
    if not MATPLOTLIB_OK or not weight_table: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    subs   = sorted(weight_table.keys())
    n_ch   = max(v.shape[0] for v in weight_table.values())
    matrix = np.full((n_ch, len(subs)), np.nan)
    for j, s in enumerate(subs):
        w = weight_table[s]; matrix[:len(w), j] = w
    fig, ax = plt.subplots(figsize=(max(8, len(subs) * 0.9), 12))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto",
                   vmin=0.1, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="channel weight")
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([s.replace("sub-", "Sub ") for s in subs], fontsize=9)
    ax.set_yticks(range(0, n_ch, 4))
    ax.set_yticklabels([f"Ch {i+1}" for i in range(0, n_ch, 4)], fontsize=7)
    ax.set_xlabel("Subject"); ax.set_ylabel("EEG Channel")
    ax.set_title("EEG Channel Weight Heatmap", fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "channel_weight_heatmap.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def plot_cv_summary(folds, fold_accs, plot_dir):
    if not MATPLOTLIB_OK or len(fold_accs) < 2: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Fold {f}" for f in folds], fold_accs, color="#2980b9", edgecolor="white")
    ax.axhline(1 / N_CLASSES, ls="--", color="gray", linewidth=1.2, label="Chance")
    mean_acc = np.mean(fold_accs)
    ax.axhline(mean_acc, ls="-", color="#e74c3c", linewidth=1.5,
               label=f"Mean = {mean_acc:.3f}")
    ax.set_ylabel("Top-1 Accuracy"); ax.set_ylim(0, max(fold_accs) * 1.25)
    ax.set_title("Cross-Validation Accuracy per Fold", fontweight="bold")
    ax.legend()
    for i, a in enumerate(fold_accs):
        ax.text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "cv_summary.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-class summary
# ──────────────────────────────────────────────────────────────────────────────

def print_per_class_summary(fold_results, label_map):
    idx_to_label  = {v: k for k, v in label_map.items()}
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for preds, labels in fold_results:
        for p, l in zip(preds, labels):
            class_total[l]   += 1
            class_correct[l] += int(p == l)
    rows = []
    for cls_idx in range(N_CLASSES):
        lbl  = idx_to_label.get(cls_idx, str(cls_idx))
        tot  = class_total[cls_idx]; corr = class_correct[cls_idx]
        rows.append((lbl, corr, tot, corr / tot if tot > 0 else 0.0))
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\n{'='*60}")
    print(f"  Per-Class Accuracy  (pooled across all folds)")
    print(f"{'='*60}")
    print(f"  {'Stimulus':<32}  Correct / Total   Acc")
    print(f"  {'-'*32}  ---------------   -----")
    for lbl, corr, tot, acc in rows:
        print(f"  {lbl:<32}  {corr:>4} / {tot:<5}       {acc:.3f}  "
              f"{'█' * int(acc * 20)}")
    all_correct = sum(r[1] for r in rows); all_total = max(1, sum(r[2] for r in rows))
    print(f"  {'-'*32}  ---------------   -----")
    print(f"  {'OVERALL':<32}  {all_correct:>4} / {all_total:<5}       "
          f"{all_correct/all_total:.3f}")
    print(f"{'='*60}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main

# ──────────────────────────────────────────────────────────────────────────────
# Main  (one full CV run PER fNIRS-configuration group → separate models)
# ──────────────────────────────────────────────────────────────────────────────

def run_group(group_name, group_subs, stimuli_dir, weight_table, args):
    group_plot_dir = PLOT_DIR / group_name
    valid_subs     = [f"sub-{i:02d}" for i in sorted(group_subs)]

    print(f"\n{'#'*60}")
    print(f"  GROUP {group_name}   subjects = {valid_subs}")
    print(f"{'#'*60}")

    group_weights = {k: v for k, v in weight_table.items()
                     if int(k.split('-')[1]) in group_subs}
    if group_weights:
        plot_channel_weight_heatmap(group_weights, group_plot_dir)

    folds          = [args.fold] if args.fold else list(range(1, N_INSTANCES + 1))
    fold_accs      = []
    fold_results   = []
    done_folds     = []
    last_label_map = None

    for fold in folds:
        tqdm.write(f"\n── [{group_name}] Fold {fold}  [test=instance{fold}] ──")
        acc, preds, labels, label_map, best_state, n_channels, curves = run_fold(
            fold, stimuli_dir, weight_table, group_subs, args)
        if acc is None:
            continue

        fold_accs.append(acc)
        fold_results.append((preds, labels))
        done_folds.append(fold)
        last_label_map = label_map

        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in range(N_CLASSES)]
        tqdm.write(f"\n  [{group_name}] Fold {fold} top-1 acc: {acc:.3f}  "
                   f"(chance = {1/N_CLASSES:.3f})")
        tqdm.write(classification_report(labels, preds, target_names=target_names,
                                         digits=3, zero_division=0))
        if curves:
            plot_training_curves(fold, curves, acc, group_plot_dir)
        plot_confusion_matrix(labels, preds, label_map, fold, group_plot_dir)

        ckpt = f"{CKPT_PREFIX}_{group_name}_fold{fold}.pt"
        torch.save({"model_state": best_state, "label_map": label_map,
                    "n_channels": n_channels, "window_sec": args.window_sec,
                    "resample_hz": RESAMPLE_FREQ, "group": group_name,
                    "group_subjects": sorted(group_subs)}, ckpt)
        tqdm.write(f"  Checkpoint → {ckpt}")

    if fold_accs:
        print(f"\n{'='*60}")
        print(f"  [{group_name}] Cross-Validation Summary")
        print(f"{'='*60}")
        for f, a in zip(done_folds, fold_accs):
            print(f"  Fold {f}: {a:.3f}  {'█' * int(a * 40)}")
        if len(fold_accs) > 1:
            arr = np.array(fold_accs)
            print(f"  {'─'*40}")
            print(f"  Mean ± Std : {arr.mean():.3f} ± {arr.std():.3f}")
            print(f"  Chance     : {1/N_CLASSES:.3f}  ({1/N_CLASSES*100:.1f}%)")
        print(f"{'='*60}")
        plot_cv_summary(done_folds, fold_accs, group_plot_dir)

    if fold_results and last_label_map:
        rows = print_per_class_summary(fold_results, last_label_map)
        plot_per_class_accuracy(rows, group_plot_dir,
                                title=f"Per-Class Accuracy — {group_name}")

    return done_folds, fold_accs


def main(args):
    stimuli_dir = Path(args.stimuli_dir)
    if not stimuli_dir.exists():
        sys.exit(f"Not found: {stimuli_dir}")

    union_allowed = GROUP_A_SUBJECTS | GROUP_B_SUBJECTS
    weights_path  = Path(args.weights_csv)
    weight_table  = {}
    if weights_path.exists():
        weight_table = load_channel_weights(weights_path, union_allowed)
    else:
        tqdm.write(f"  [WEIGHTS] WARNING: {weights_path} not found — uniform weights.")

    print(f"\n{'='*60}")
    print(f"  ENIGMA EEG Classifier  {MODEL_TAG}")
    print(f"{'='*60}")
    print(f"  Device          : {DEVICE}")
    print(f"  Stimuli dir     : {stimuli_dir}")
    print(f"  Excluded subs   : {sorted(EXCLUDED_SUBJECTS)}")
    print(f"  Group A subs    : {sorted(GROUP_A_SUBJECTS)} (fNIRS config A)")
    print(f"  Group B subs    : {sorted(GROUP_B_SUBJECTS)} (fNIRS config B)")
    print(f"  Weights CSV     : {weights_path} "
          f"({'loaded' if weight_table else 'NOT FOUND'})")
    print(f"  Resampled Hz    : {RESAMPLE_FREQ}")
    print(f"  Window/Overlap  : {args.window_sec}s / {args.overlap_sec}s")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Cache dir       : {'disabled' if args.no_cache else str(CACHE_DIR)}")
    print(f"  Plot dir        : {PLOT_DIR}")
    print(f"  Folds           : {args.fold if args.fold else '1-5 (full CV)'}")
    print(f"{'='*60}\n")

    summary = {}
    for group_name, group_subs in GROUPS:
        summary[group_name] = run_group(group_name, group_subs,
                                        stimuli_dir, weight_table, args)

    print(f"\n{'='*60}")
    print(f"  COMBINED SUMMARY ACROSS fNIRS GROUPS")
    print(f"{'='*60}")
    for group_name, (done_folds, fold_accs) in summary.items():
        if fold_accs:
            m = np.mean(fold_accs)
            print(f"  {group_name:<20}  mean acc = {m:.3f}  over folds {done_folds}")
        else:
            print(f"  {group_name:<20}  no usable data")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ENIGMA EEG {MODEL_TAG}")
    parser.add_argument("--stimuli_dir",     default="stimuli",  type=str)
    parser.add_argument("--weights_csv",
                        default="eeg_channel_weights.csv",
                        type=str)
    parser.add_argument("--fold",            default=None,  type=int)
    parser.add_argument("--window_sec",      default=1.0,   type=float)
    parser.add_argument("--overlap_sec",     default=0.5,   type=float)
    parser.add_argument("--pretrain_epochs", default=100,   type=int)
    parser.add_argument("--epochs",          default=200,   type=int)
    parser.add_argument("--batch_size",      default=96,    type=int)
    parser.add_argument("--lr",              default=5e-4,  type=float)
    parser.add_argument("--no_cache",        action="store_true")
    args = parser.parse_args()
    main(args)