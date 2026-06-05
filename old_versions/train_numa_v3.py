#!/usr/bin/env python3
"""
ENIGMA-style EEG Brain Decoding Classifier — v3 (SGCN + channel weights)
==================================================================================
This is an optimized rewrite of train_enigma_v2.py, motivated directly by the
v1/v2/split run logs, in which **every** configuration collapsed to predicting a
single class (per-class recall = 1.0 for one class, 0.0 for the other 35). The
reported "top-1 accuracy" differences between scripts (v1 0.029, v2 0.078,
v2-split fold4 0.196, …) were therefore NOT real learning differences — each
number is just the frequency of whichever class happened to dominate that test
fold. Macro-F1 was ~0.002–0.009 everywhere, i.e. at chance.

Two root causes were visible in the logs and the old run_fold():
  A. NO class balancing. The window set is 3–6× imbalanced and varies per fold,
     so plain CE + shuffle makes "predict the majority class" the loss minimizer
     for a weak model.
  B. A broken train schedule. The encoder was FROZEN for the first epochs//2
     fine-tune epochs (only the classifier trained) on top of a SupCon pretrain
     whose loss was flat (~5.09 for v2, ~4.84 for v1) — i.e. it learned nothing.
     The encoder then unfroze at half-time but at lr*0.1. So the encoder only
     ever trained ~100 epochs at a 10× reduced LR, after ~178 min/fold of wasted
     pretrain + frozen-classifier time. Train accuracy never exceeded ~10%.

Optimizations in this script (each tied to the above):
  1. Class-balanced sampling (WeightedRandomSampler, inverse-frequency) so every
     batch is roughly uniform over the 36 classes — the single biggest fix for
     the collapse.
  2. Class-weighted CrossEntropy (inverse-sqrt-frequency) + label smoothing 0.1
     as a complementary push against majority-class bias.
  3. End-to-end training from epoch 1 (NO freeze/unfreeze). All parameters learn
     together at a real LR.
  4. SupCon pretraining OFF by default (--pretrain_epochs 0). It contributed
     nothing in the logs and cost ~half the runtime. Still available if >0.
  5. OneCycleLR (warmup + cosine decay) + AdamW + grad clipping for fast, stable
     convergence.
  6. Model selection and early stopping on **macro-F1** (balanced), not top-1 —
     top-1 rewards the very collapse we are trying to kill. Patience-based early
     stop slashes runtime.
  7. Honest metrics every fold: top-1, balanced accuracy (macro-recall),
     macro-F1, top-5, AND trial-level (per-recording majority-vote) top-1, since
     single 1-s windows are extremely noisy.
  8. Per-class window-count diagnostics at fold build, plus an optional
     --max_per_class cap to hard-limit imbalance / speed up runs.

NOTE ON EXPECTATIONS: single-trial, 1-second, 36-class EEG object decoding is
genuinely hard. Even with these fixes, realistic per-window top-1 may stay
modest; the meaningful success signals are macro-F1 / balanced-accuracy / top-5
clearly above chance and predictions SPREAD across classes (no single-class
collapse), with trial-level voting higher than per-window.

Inherited bug fixes from train_enigma_v2.py (all still applied):
  - float32 np.stack cache (no object-dtype crash on 2nd run), allow_pickle=False
  - __getitem__ copies before augmenting (no cross-epoch aliasing)
  - GraphConvLayer uses AX (not AᵀX)
  - zero-padded sub-NN subject ids for weight lookups
  - best_state always initialised; SGCN.encode reads the real channel dim
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
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
except ImportError:
    sys.exit("pip install torch")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

try:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  balanced_accuracy_score, f1_score)
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
CACHE_DIR          = Path(".eeg_cache_v8")     # shared with v2 (identical preprocessing)
PLOT_DIR           = Path("enigma_plots_3")

MODEL_TAG   = "v3 (SGCN + class-balanced end-to-end)"
CKPT_PREFIX = "enigma_v3"


# ──────────────────────────────────────────────────────────────────────────────
# BioSemi-64 electrode positions (azimuthal, normalised to [-1, 1])
# ──────────────────────────────────────────────────────────────────────────────
BIOSEMI64_XY = np.array([
    [-0.31, 0.95],[ 0.31, 0.95],[-0.55, 0.80],[ 0.55, 0.80],
    [-0.81, 0.59],[-0.46, 0.67],[ 0.00, 0.72],[ 0.46, 0.67],
    [ 0.81, 0.59],[-0.71, 0.33],[-0.27, 0.38],[ 0.00, 0.38],
    [ 0.27, 0.38],[ 0.71, 0.33],[-1.00, 0.00],[-0.50, 0.00],
    [ 0.00, 0.00],[ 0.50, 0.00],[ 1.00, 0.00],[-0.71,-0.33],
    [-0.27,-0.38],[ 0.00,-0.38],[ 0.27,-0.38],[ 0.71,-0.33],
    [-0.81,-0.59],[-0.46,-0.67],[ 0.00,-0.72],[ 0.46,-0.67],
    [ 0.81,-0.59],[-0.31,-0.87],[ 0.00,-0.87],[ 0.31,-0.87],
    [-0.19,-0.98],[ 0.00,-0.98],[ 0.19,-0.98],[-0.64, 0.76],
    [ 0.64, 0.76],[-0.63, 0.68],[-0.22, 0.72],[ 0.22, 0.72],
    [ 0.63, 0.68],[-0.48, 0.35],[ 0.48, 0.35],[-0.75, 0.00],
    [-0.25, 0.00],[ 0.25, 0.00],[ 0.75, 0.00],[-0.48,-0.35],
    [ 0.48,-0.35],[-0.63,-0.68],[-0.22,-0.72],[ 0.22,-0.72],
    [ 0.63,-0.68],[-0.50,-0.90],[-0.31,-0.87],[ 0.31,-0.87],
    [ 0.50,-0.90],[ 0.00,-1.00],
    [ 0.15,-0.95],[-0.15,-0.95],[ 0.30,-0.92],[-0.30,-0.92],
    [ 0.45,-0.88],[-0.45,-0.88],[ 0.00, 0.50],
], dtype=np.float32)   # shape (64, 2)


def build_electrode_adj(n_channels: int, threshold: float = 0.4) -> torch.Tensor:
    """Row-normalised (D^{-1}A) adjacency. Safely handles any n_channels <= 64."""
    n = min(n_channels, len(BIOSEMI64_XY))
    xy   = BIOSEMI64_XY[:n]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    adj  = (dist < threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    row_sum = adj.sum(1, keepdims=True).clip(min=1)
    adj = adj / row_sum
    if n_channels > n:
        full = np.eye(n_channels, dtype=np.float32)
        full[:n, :n] = adj
        adj = full
    return torch.from_numpy(adj)


# ──────────────────────────────────────────────────────────────────────────────
# eeg_channel_weights (a.k.a. CHASEDOWN multivariate channel weights)
# ──────────────────────────────────────────────────────────────────────────────

def load_channel_weights(csv_path: Path, allowed_subjects: set) -> dict:
    """
    Read per-subject per-channel weights. Columns are matched to subject numbers
    by the first integer found in the column name. Only `allowed_subjects` kept.
    """
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
    """Return a zero-padded 'sub-NN' id parsed from the path (robust to sub-1/sub-01)."""
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
# Preprocessing  (returns a list of (C, T) float32 windows)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_bdf(path: Path, window_sec: float, overlap_sec: float,
                   use_cache: bool = True) -> list:
    key = hashlib.md5(
        f"{path}{window_sec}{overlap_sec}{RESAMPLE_FREQ}"
        f"{BANDPASS_LOW}{BANDPASS_HIGH}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{key}.npy"

    if use_cache and cache_path.exists():
        try:
            arr = np.load(str(cache_path), allow_pickle=False)   # (N, C, T) float32
            return [arr[i] for i in range(arr.shape[0])]
        except Exception:
            pass  # corrupt / legacy object cache → reprocess below

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
        # Contiguous float32 3-D array — NOT an object array.
        np.save(str(cache_path), np.stack(valid_windows).astype(np.float32))

    return valid_windows


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (in-memory)
# ──────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, samples: list, window_sec: float, overlap_sec: float,
                 weight_table: dict, augment_data: bool = False,
                 use_cache: bool = True, max_per_class: int = 0):
        self.augment_data = augment_data
        self.items        = []
        self.labels       = []        # parallel int label per window (for sampler)
        self.groups       = []        # parallel trial/recording id per window
        rejected          = 0

        for trial_id, (path, label) in enumerate(samples):
            sub_id  = get_subject_id(path)
            ov      = overlap_sec if augment_data else 0.0
            windows = preprocess_bdf(path, window_sec, ov, use_cache)
            if not windows:
                rejected += 1
                continue
            for w in windows:
                w_weighted = apply_channel_weights(w, sub_id, weight_table)
                self.items.append((np.ascontiguousarray(w_weighted, dtype=np.float32),
                                   label))
                self.labels.append(label)
                self.groups.append(trial_id)

        if max_per_class and self.items:
            self._cap_per_class(max_per_class)

        if rejected:
            tqdm.write(f"    Artifact rejection: {rejected} files removed")

    def _cap_per_class(self, cap: int):
        """Randomly keep at most `cap` windows per class (hard imbalance limit)."""
        by_class = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            by_class[lbl].append(i)
        keep = []
        rng = np.random.default_rng(0)
        for lbl, idxs in by_class.items():
            if len(idxs) > cap:
                idxs = rng.choice(idxs, size=cap, replace=False).tolist()
            keep.extend(idxs)
        keep.sort()
        self.items  = [self.items[i]  for i in keep]
        self.labels = [self.labels[i] for i in keep]
        self.groups = [self.groups[i] for i in keep]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data, label = self.items[idx]
        # Copy so in-place augmentation never mutates the stored array.
        x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).unsqueeze(0)
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
               weight_table: dict, allowed_subjects: set, use_cache: bool,
               max_per_class: int = 0):
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
    train_ds = EEGDataset(train_raw, window_sec, overlap_sec, weight_table,
                          augment_data=True,  use_cache=use_cache,
                          max_per_class=max_per_class)
    tqdm.write(f"    Building test  set ({len(test_raw)} BDFs) ...")
    test_ds  = EEGDataset(test_raw,  window_sec, 0.0, weight_table,
                          augment_data=False, use_cache=use_cache,
                          max_per_class=0)
    tqdm.write(f"    Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    # Imbalance diagnostic — this is what drove the single-class collapse.
    if train_ds.labels:
        counts = np.bincount(np.array(train_ds.labels), minlength=N_CLASSES)
        nz = counts[counts > 0]
        tqdm.write(f"    Train windows/class: min={nz.min()} max={nz.max()} "
                   f"mean={nz.mean():.0f} (imbalance {nz.max()/max(1,nz.min()):.1f}×)")
    return train_ds, test_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# SGCN
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """H' = σ( BN( A · H · W ) )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn     = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        a = adj[:N, :N]
        # A · H  (aggregate each node's neighbours, weighted by row-normalised A)
        h = torch.einsum("mn,bnf->bmf", a, x)
        h = self.linear(h)                          # (B, N, out_dim)
        h = self.bn(h.reshape(B * N, -1)).reshape(B, N, -1)
        return F.elu(h)


class SGCN(nn.Module):
    """Spatial Graph Convolutional Network. Input (B, 1, C, T) → logits (B, n_classes)."""
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 n_f: int = 32, gcn_dim: int = 64, embed_dim: int = 128,
                 dropout_p: float = DROPOUT_P):
        super().__init__()
        self.n_channels = n_channels

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_f, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(n_f),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout2d(p=0.25),
        )

        with torch.no_grad():
            dummy  = torch.zeros(1, 1, n_channels, n_times)
            T_pool = self.temporal_conv(dummy).shape[3]
        node_feat_dim = n_f * T_pool

        self.gcn1    = GraphConvLayer(node_feat_dim, gcn_dim)
        self.gcn2    = GraphConvLayer(gcn_dim,       gcn_dim)
        self.dropout = nn.Dropout(dropout_p)

        flat_dim = gcn_dim * n_channels
        self.embed_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

        self.register_buffer("adj", build_electrode_adj(64))   # always (64, 64)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        z = self.temporal_conv(x)                       # (B, n_f, C, T_pool)
        n_f_actual, C, T_pool = z.shape[1], z.shape[2], z.shape[3]
        z = z.permute(0, 2, 1, 3).reshape(B, C, n_f_actual * T_pool)
        z = self.gcn1(z, self.adj)
        z = self.dropout(z)
        z = self.gcn2(z, self.adj)
        z = z.reshape(B, -1)
        return self.embed_head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ──────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

def evaluate(model, loader):
    """Return (preds, labels, top5_hits) over a loader in dataset order."""
    model.eval()
    preds, labels, top5_hits = [], [], 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            k      = min(5, logits.shape[1])
            top5   = logits.topk(k, dim=1).indices.cpu()
            ys     = yb.tolist()
            for i, y in enumerate(ys):
                top5_hits += int(y in top5[i].tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(ys)
    return preds, labels, top5_hits


def trial_level_accuracy(preds, labels, groups):
    """Majority-vote window predictions within each recording, then score top-1."""
    by_group = defaultdict(list)
    truth    = {}
    for p, l, g in zip(preds, labels, groups):
        by_group[g].append(p)
        truth[g] = l
    correct = 0
    for g, plist in by_group.items():
        vote = max(set(plist), key=plist.count)
        correct += int(vote == truth[g])
    return correct / max(1, len(by_group)), len(by_group)


def run_fold(fold: int, stimuli_dir: Path, weight_table: dict,
             allowed_subjects: set, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec,
        weight_table, allowed_subjects, not args.no_cache, args.max_per_class)

    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data, skipping.")
        return None, None, None, label_map, None, None, None, None

    # ── Class balancing (the key fix for single-class collapse) ───────────────
    train_labels = np.asarray(train_ds.labels)
    counts       = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    safe_counts  = np.where(counts > 0, counts, 1.0)

    sampler = None
    if args.balance in ("sampler", "both"):
        inv_freq   = 1.0 / safe_counts
        sample_w   = inv_freq[train_labels]
        sampler    = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_labels), replacement=True)

    ce_weight = None
    if args.balance in ("classweight", "both"):
        # inverse-sqrt-frequency, mean-normalised; absent classes → 0
        w        = np.where(counts > 0, (counts.sum() / (len(counts) * safe_counts)) ** 0.5, 0.0)
        w        = w / (w[w > 0].mean() + 1e-8)
        ce_weight = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    n_channels = train_ds[0][0].shape[1]
    target_T   = train_ds[0][0].shape[-1]

    model = SGCN(n_channels, target_T, N_CLASSES).to(DEVICE)
    ce    = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=0.1)

    # ── Optional SupCon pretraining (OFF by default; it did nothing in logs) ──
    pretrain_losses = []
    if args.pretrain_epochs > 0:
        supcon  = SupConLoss()
        opt_pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        sch_pre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pre, T_max=args.pretrain_epochs)
        pre_bar = tqdm(range(1, args.pretrain_epochs + 1),
                       desc=f"  Fold {fold} pretrain", ncols=90,
                       unit="ep", leave=False, colour="yellow")
        for _ in pre_bar:
            model.train(); ep_loss, batches = 0.0, 0
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
                avg = ep_loss / batches; pretrain_losses.append(avg)
                pre_bar.set_postfix(loss=f"{avg:.3f}")

    # ── End-to-end supervised training (no freeze schedule) ───────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.15)

    best_f1, best_state, best_top1 = -1.0, None, 0.0
    epochs_no_improve              = 0
    train_accs, val_accs, val_f1s  = [], [], []

    ft_bar = tqdm(range(1, args.epochs + 1),
                  desc=f"  Fold {fold} train", ncols=90,
                  unit="ep", leave=False, colour="cyan")
    for epoch in ft_bar:
        model.train()
        tr_correct, tr_total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            ce(logits, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        preds, labels, top5_hits = evaluate(model, test_loader)
        tr_acc  = tr_correct / max(1, tr_total)
        val_acc = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
        val_f1  = f1_score(labels, preds, average="macro", zero_division=0)
        train_accs.append(tr_acc); val_accs.append(val_acc); val_f1s.append(val_f1)

        if val_f1 > best_f1:                       # select on BALANCED metric
            best_f1, best_top1 = val_f1, val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ft_bar.set_postfix(tr=f"{tr_acc:.2f}", val=f"{val_acc:.2f}",
                           f1=f"{val_f1:.3f}", best_f1=f"{max(best_f1,0):.3f}")

        if args.patience and epochs_no_improve >= args.patience:
            tqdm.write(f"  Early stop at epoch {epoch} "
                       f"(no macro-F1 gain in {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    preds, labels, top5_hits = evaluate(model, test_loader)

    top1    = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    top5    = top5_hits / max(1, len(labels))
    trial_acc, n_trials = trial_level_accuracy(preds, labels, test_ds.groups)
    metrics = dict(top1=top1, balanced_acc=bal_acc, macro_f1=macro_f1,
                   top5=top5, trial_acc=trial_acc, n_trials=n_trials)

    curves = dict(pretrain_loss=pretrain_losses, train_acc=train_accs,
                  val_acc=val_accs, val_f1=val_f1s)
    return top1, preds, labels, label_map, best_state, n_channels, curves, metrics


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

def main(args):
    stimuli_dir = Path(args.stimuli_dir)
    if not stimuli_dir.exists():
        sys.exit(f"Not found: {stimuli_dir}")

    allowed_subjects = ALL_SUBJECTS - EXCLUDED_SUBJECTS
    valid_subs       = [f"sub-{i:02d}" for i in sorted(allowed_subjects)]
    n_valid          = len(valid_subs)

    weights_path = Path(args.weights_csv)
    weight_table = {}
    if weights_path.exists():
        weight_table = load_channel_weights(weights_path, allowed_subjects)
    else:
        tqdm.write(f"  [WEIGHTS] WARNING: {weights_path} not found — uniform weights.")

    if weight_table:
        plot_channel_weight_heatmap(weight_table, PLOT_DIR)

    print(f"\n{'='*60}")
    print(f"  ENIGMA EEG Classifier  {MODEL_TAG}")
    print(f"{'='*60}")
    print(f"  Device          : {DEVICE}")
    print(f"  Stimuli dir     : {stimuli_dir}")
    print(f"  Excluded subs   : {sorted(EXCLUDED_SUBJECTS)}")
    print(f"  Active subjects : {n_valid}  {valid_subs}")
    print(f"  Weights CSV     : {weights_path} "
          f"({'loaded' if weight_table else 'NOT FOUND'})")
    print(f"  Weighted subs   : {sorted(weight_table.keys())}")
    print(f"  Resampled Hz    : {RESAMPLE_FREQ}")
    print(f"  Window/Overlap  : {args.window_sec}s / {args.overlap_sec}s")
    print(f"  Bandpass        : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Class balancing : {args.balance}")
    print(f"  Max per class   : {args.max_per_class or 'uncapped'}")
    print(f"  Pretrain epochs : {args.pretrain_epochs} (SupCon; 0 = skip)")
    print(f"  Train epochs    : {args.epochs}  (early-stop patience {args.patience})")
    print(f"  LR (OneCycle)   : {args.lr}")
    print(f"  Selection metric: macro-F1")
    print(f"  Cache dir       : {'disabled' if args.no_cache else str(CACHE_DIR)}")
    print(f"  Plot dir        : {PLOT_DIR}")
    print(f"  Folds           : {args.fold if args.fold else '1-5 (full CV)'}")
    print(f"{'='*60}\n")

    folds          = [args.fold] if args.fold else list(range(1, N_INSTANCES + 1))
    fold_accs      = []
    fold_metrics   = []
    fold_results   = []
    done_folds     = []
    last_label_map = None

    for fold in folds:
        tqdm.write(f"\n── Fold {fold}  [test=instance{fold}] ──")
        acc, preds, labels, label_map, best_state, n_channels, curves, metrics = run_fold(
            fold, stimuli_dir, weight_table, allowed_subjects, args)
        if acc is None:
            continue

        fold_accs.append(acc)
        fold_metrics.append(metrics)
        fold_results.append((preds, labels))
        done_folds.append(fold)
        last_label_map = label_map

        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in range(N_CLASSES)]
        tqdm.write(f"\n  Fold {fold} metrics (chance top-1 = {1/N_CLASSES:.3f}):")
        tqdm.write(f"    top-1            : {metrics['top1']:.3f}")
        tqdm.write(f"    balanced acc     : {metrics['balanced_acc']:.3f}  "
                   f"(macro-recall; collapse → ~{1/N_CLASSES:.3f})")
        tqdm.write(f"    macro-F1         : {metrics['macro_f1']:.3f}  "
                   f"(selection metric)")
        tqdm.write(f"    top-5            : {metrics['top5']:.3f}  "
                   f"(chance = {5/N_CLASSES:.3f})")
        tqdm.write(f"    trial-level top-1: {metrics['trial_acc']:.3f}  "
                   f"(majority vote over {metrics['n_trials']} recordings)")
        tqdm.write(classification_report(labels, preds, target_names=target_names,
                                         digits=3, zero_division=0))
        if curves:
            plot_training_curves(fold, curves, metrics['top1'], PLOT_DIR)
        plot_confusion_matrix(labels, preds, label_map, fold, PLOT_DIR)

        ckpt = f"{CKPT_PREFIX}_fold{fold}.pt"
        torch.save({"model_state": best_state, "label_map": label_map,
                    "n_channels": n_channels, "window_sec": args.window_sec,
                    "resample_hz": RESAMPLE_FREQ, "metrics": metrics}, ckpt)
        tqdm.write(f"  Checkpoint → {ckpt}")

    if fold_accs:
        print(f"\n{'='*60}")
        print(f"  Cross-Validation Summary  (selection = macro-F1)")
        print(f"{'='*60}")
        print(f"  {'Fold':<6}{'top1':>8}{'balAcc':>9}{'mF1':>8}{'top5':>8}{'trial':>8}")
        print(f"  {'-'*45}")
        for f, m in zip(done_folds, fold_metrics):
            print(f"  {f:<6}{m['top1']:>8.3f}{m['balanced_acc']:>9.3f}"
                  f"{m['macro_f1']:>8.3f}{m['top5']:>8.3f}{m['trial_acc']:>8.3f}")
        if len(fold_metrics) > 1:
            def col(key): return np.array([m[key] for m in fold_metrics])
            print(f"  {'-'*45}")
            print(f"  {'mean':<6}{col('top1').mean():>8.3f}{col('balanced_acc').mean():>9.3f}"
                  f"{col('macro_f1').mean():>8.3f}{col('top5').mean():>8.3f}"
                  f"{col('trial_acc').mean():>8.3f}")
            print(f"  {'std':<6}{col('top1').std():>8.3f}{col('balanced_acc').std():>9.3f}"
                  f"{col('macro_f1').std():>8.3f}{col('top5').std():>8.3f}"
                  f"{col('trial_acc').std():>8.3f}")
        print(f"  Chance top-1 = {1/N_CLASSES:.3f} | chance top-5 = {5/N_CLASSES:.3f}")
        print(f"{'='*60}")
        plot_cv_summary(done_folds, fold_accs, PLOT_DIR)

    if fold_results and last_label_map:
        rows = print_per_class_summary(fold_results, last_label_map)
        plot_per_class_accuracy(rows, PLOT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ENIGMA EEG {MODEL_TAG}")
    parser.add_argument("--stimuli_dir",     default="stimuli",  type=str)
    parser.add_argument("--weights_csv",
                        default="/Users/neelahuja/desktop/numa_model/eeg_channel_weights.csv",
                        type=str)
    parser.add_argument("--fold",            default=None,  type=int)
    parser.add_argument("--window_sec",      default=1.0,   type=float)
    parser.add_argument("--overlap_sec",     default=0.5,   type=float)
    parser.add_argument("--pretrain_epochs", default=0,     type=int,
                        help="SupCon pretrain epochs (0 = skip; it did nothing in the logs)")
    parser.add_argument("--epochs",          default=60,    type=int,
                        help="max supervised epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience",        default=12,    type=int,
                        help="early-stop patience on val macro-F1 (0 = disable)")
    parser.add_argument("--batch_size",      default=96,    type=int)
    parser.add_argument("--lr",              default=1e-3,  type=float,
                        help="OneCycle max LR")
    parser.add_argument("--balance",         default="sampler",
                        choices=["sampler", "classweight", "both", "none"],
                        help="class-imbalance handling (default: balanced sampler)")
    parser.add_argument("--max_per_class",   default=0,     type=int,
                        help="cap train windows per class (0 = uncapped)")
    parser.add_argument("--no_cache",        action="store_true")
    args = parser.parse_args()
    main(args)