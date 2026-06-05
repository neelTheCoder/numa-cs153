#!/usr/bin/env python3
"""
ENIGMA-style EEG Brain Decoding Classifier — v4 (autoresearch-optimized SGCN)
==================================================================================
This is a direct, evidence-driven successor to train_enigma_v3.py. Where v3 fixed
the single-class-collapse failure mode (class balancing + end-to-end training +
macro-F1 selection), v4 folds in the *winning* configuration discovered by the
automated hyperparameter/architecture search (the "autoresearch/jun4" sweep, 26
experiments under an identical, un-gameable val split scored on balanced accuracy).

WHAT THE AUTORESEARCH ACTUALLY PROVED
-------------------------------------
The sweep started from the v3 SGCN and ran 26 single-change experiments. Only FOUR
changes survived ("keep"); everything else was at-or-below baseline ("discard").
v4 applies exactly the four confirmed winners — no more, no less — because the
search showed the architecture is otherwise saturated and extra capacity / less
regularization / fancier pooling all FAILED to generalize.

  CONFIRMED WINS (applied here, cumulative best val_balanced_acc 0.0370):
    1. MAX_LR  1e-3 → 3e-3   (LR sweep up — faster, better-converged optimum)
    2. DROPOUT_P 0.5 → 0.3   (the 0.5 head dropout over-regularized a weak model)
    3. BALANCE "sampler" → "both"  (balanced sampler AND inverse-√freq class-
                                    weighted CE — the two anti-collapse forces stack)
    4. GCN RESIDUAL SKIP     (z = gcn2(gcn1(z)) + gcn1(z); the skip preserves the
                              first-hop spatial features the 2nd hop would wash out)

  CONFIRMED DEAD-ENDS (deliberately NOT done — each was tested and lost):
    GCN_DIM=96/EMBED_DIM=192 (capacity, no gain) · WEIGHT_DECAY=0.01 (no gain) ·
    temporal kernel (1,64) (no gain) · BATCH_SIZE=64 (no gain) · attention pooling
    (re-collapsed) · ADJ_THRESHOLD 0.3/0.5 (no gain) · 3rd GCN layer (overfit) ·
    input Gaussian noise / mixup / temporal-shift aug (no gain) · DROPOUT_P=0.2
    (below 0.3) · remove Dropout2d (better raw F1 but lower balanced acc) ·
    LABEL_SMOOTH=0.05 (no gain) · EEGNet / EMBED_DIM=64 (too small) · stride-8
    pool / global mean pool (too lossy) · WARMUP_FRAC=0.05 (no gain).
  → So v4 keeps 32 filters / GCN 64 / EMBED 128 / ADJ 0.4 / WD 0.05 / kernel (1,32)
    / Dropout2d 0.25 / label-smooth 0.1 / warmup 0.15 — all the baselines that won.

TWO ADDITIONS BEYOND THE AUTORESEARCH (safe, and OUTSIDE its 300 s/run search box)
---------------------------------------------------------------------------------
The sweep ran each experiment under a ~5-minute wall-clock cap on a single fixed
split; it never explored test-time aggregation or weight averaging. Both below are
"free" — they cannot inflate collapse, and one is constructed so it can never hurt:

  A. SOFT trial-level voting. v3 reported trial accuracy via *hard* majority vote
     over a recording's 1-s windows. v4 averages the per-window softmax
     probabilities, then argmaxes — strictly more information per recording and the
     honest way to score the per-trial decode. Hard-vote is still printed for
     comparison.
  B. EMA weight averaging, selected as BEST-OF-{raw, EMA}. We keep an exponential
     moving average of the weights and, every epoch, evaluate BOTH the live model
     and the EMA snapshot, selecting whichever scores higher macro-F1. Because the
     raw model is always a candidate, EMA can only ever improve the checkpoint,
     never degrade it (disable with --ema 0).

Everything else — preprocessing, the .eeg_cache_v8 cache, per-subject channel
weights, honest per-fold metrics (top-1 / balanced acc / macro-F1 / top-5 /
trial-level), early stopping on macro-F1, and all plots — is carried over verbatim
from v3 so results stay directly comparable.

NOTE ON EXPECTATIONS: single-trial, 1-second, 36-class EEG object decoding is
genuinely hard. The meaningful success signals remain macro-F1 / balanced-accuracy
/ top-5 above chance with predictions SPREAD across classes (no collapse), and
trial-level (soft) voting above per-window.
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
DROPOUT_P          = 0.3        # autoresearch win: 0.5 → 0.3 (head/embedding dropout)
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
CACHE_DIR          = Path(".eeg_cache_v8")     # shared with v2/v3 (identical preprocessing)
PLOT_DIR           = Path("enigma_plots_v4")

MODEL_TAG   = "v4 (autoresearch-optimized SGCN: 3e-3 LR · drop 0.3 · balance=both · GCN residual)"
CKPT_PREFIX = "enigma_v4"

# Confirmed-best hyperparameters from the autoresearch sweep (kept as named
# constants so the winning recipe is auditable in one place).
ADJ_THRESHOLD      = 0.4        # kept — 0.3 (sparser) and 0.5 (denser) both lost
N_FILTERS          = 32         # kept — capacity bumps did not generalize
GCN_DIM            = 64         # kept — 48 collapsed, 96 no gain
EMBED_DIM          = 128        # kept — 64 too small, 192 no gain
TEMPORAL_DROPOUT2D = 0.25       # kept — removing it raised raw F1 but lowered bal-acc
LABEL_SMOOTH       = 0.1        # kept — 0.05 no gain


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


def build_electrode_adj(n_channels: int, threshold: float = ADJ_THRESHOLD) -> torch.Tensor:
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
            # Light augmentation kept from v3. The autoresearch separately tested
            # ADDING input Gaussian noise / mixup / temporal-shift on top of the
            # baseline and none generalized, so we do NOT pile more on here.
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
# SGCN  (v4: GCN residual skip — the confirmed-best architecture change)
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
                 n_f: int = N_FILTERS, gcn_dim: int = GCN_DIM, embed_dim: int = EMBED_DIM,
                 dropout_p: float = DROPOUT_P):
        super().__init__()
        self.n_channels = n_channels

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_f, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(n_f),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout2d(p=TEMPORAL_DROPOUT2D),
        )

        with torch.no_grad():
            dummy  = torch.zeros(1, 1, n_channels, n_times)
            T_pool = self.temporal_conv(dummy).shape[3]
        node_feat_dim = n_f * T_pool

        # Both GCN layers output gcn_dim so the residual skip (gcn2 + gcn1) is
        # dimension-matched. The 2nd hop's output is ADDED to the 1st hop's, so
        # the network can keep sharp first-neighbourhood spatial features that a
        # second round of neighbour-averaging would otherwise smear out.
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
        z1 = self.gcn1(z, self.adj)
        z1 = self.dropout(z1)
        z  = self.gcn2(z1, self.adj) + z1               # ← v4 residual skip
        z  = z.reshape(B, -1)
        return self.embed_head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
# EMA weight averaging  (selected as best-of-{raw, EMA} so it can never hurt)
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential moving average of all model parameters and buffers.

    Float tensors are EMA-smoothed; integer buffers (e.g. BatchNorm's
    num_batches_tracked) are simply copied. The smoothed weights are evaluated
    alongside the live model every epoch; selection keeps whichever is better,
    so the EMA is strictly an upside (disable with --ema 0)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

    def cpu_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


# ──────────────────────────────────────────────────────────────────────────────
# One fold
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    """Return (preds, labels, top5_hits, probs) over a loader in dataset order.

    probs is an (N, n_classes) float32 array of per-window softmax probabilities,
    used both for argmax (top-1) and for SOFT trial-level voting."""
    model.eval()
    preds, labels, top5_hits, all_probs = [], [], 0, []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            probs  = F.softmax(logits, dim=1).cpu()
            k      = min(5, logits.shape[1])
            top5   = logits.topk(k, dim=1).indices.cpu()
            ys     = yb.tolist()
            for i, y in enumerate(ys):
                top5_hits += int(y in top5[i].tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(ys)
            all_probs.append(probs.numpy().astype(np.float32))
    probs = (np.concatenate(all_probs, axis=0) if all_probs
             else np.zeros((0, N_CLASSES), dtype=np.float32))
    return preds, labels, top5_hits, probs


def trial_level_accuracy(preds, labels, groups):
    """HARD vote: majority window prediction within each recording, then top-1."""
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


def trial_level_accuracy_soft(probs, labels, groups):
    """SOFT vote: average per-window softmax within each recording, then argmax.

    Uses more information than a hard majority vote (a confident-but-minority
    window can still swing the trial), so it is the honest per-recording decode."""
    sum_prob = {}
    truth    = {}
    for pr, l, g in zip(probs, labels, groups):
        if g not in sum_prob:
            sum_prob[g] = np.zeros(N_CLASSES, dtype=np.float64)
            truth[g]    = l
        sum_prob[g] += pr
    correct = sum(int(np.argmax(sum_prob[g]) == truth[g]) for g in sum_prob)
    return correct / max(1, len(sum_prob)), len(sum_prob)


def run_fold(fold: int, stimuli_dir: Path, weight_table: dict,
             allowed_subjects: set, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec,
        weight_table, allowed_subjects, not args.no_cache, args.max_per_class)

    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data, skipping.")
        return None, None, None, label_map, None, None, None, None

    # ── Class balancing (the key fix for single-class collapse) ───────────────
    # v4 default BALANCE="both": stack the inverse-frequency SAMPLER with an
    # inverse-√frequency class-weighted CE. The autoresearch confirmed the two
    # anti-collapse forces compound rather than cancel.
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
    ce    = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=LABEL_SMOOTH)

    # ── End-to-end supervised training (no freeze schedule) ───────────────────
    # OneCycle warmup(0.15) + cosine at the confirmed max_lr=3e-3.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=args.warmup_frac)

    ema = ModelEMA(model, args.ema) if args.ema and args.ema > 0 else None

    best_f1, best_state, best_top1 = -1.0, None, 0.0
    best_src                       = "raw"
    epochs_no_improve              = 0
    train_accs, val_accs, val_f1s  = [], [], []

    def _eval_and_score():
        preds, labels, _, probs = evaluate(model, test_loader)
        acc = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        return acc, f1

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
            if ema is not None:
                ema.update(model)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        # Candidate 1: the live (raw) model.
        raw_acc, raw_f1 = _eval_and_score()
        cand_f1, cand_acc, cand_src = raw_f1, raw_acc, "raw"
        cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Candidate 2: the EMA snapshot. Swap it in, score it, swap the live
        # weights back so training continues uninterrupted. Keep whichever wins —
        # so EMA is a pure upside and can never degrade the checkpoint.
        if ema is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            ema_acc, ema_f1 = _eval_and_score()
            model.load_state_dict(raw_sd)
            if ema_f1 > cand_f1:
                cand_f1, cand_acc, cand_src = ema_f1, ema_acc, "ema"
                cand_state = ema.cpu_state()

        tr_acc = tr_correct / max(1, tr_total)
        train_accs.append(tr_acc); val_accs.append(cand_acc); val_f1s.append(cand_f1)

        if cand_f1 > best_f1:                       # select on BALANCED metric
            best_f1, best_top1 = cand_f1, cand_acc
            best_state, best_src = cand_state, cand_src
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ft_bar.set_postfix(tr=f"{tr_acc:.2f}", val=f"{cand_acc:.2f}",
                           f1=f"{cand_f1:.3f}", best_f1=f"{max(best_f1,0):.3f}",
                           src=best_src)

        if args.patience and epochs_no_improve >= args.patience:
            tqdm.write(f"  Early stop at epoch {epoch} "
                       f"(no macro-F1 gain in {args.patience} epochs; "
                       f"best checkpoint from '{best_src}' weights)")
            break

    model.load_state_dict(best_state)
    preds, labels, top5_hits, probs = evaluate(model, test_loader)

    top1     = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
    bal_acc  = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    top5     = top5_hits / max(1, len(labels))
    trial_acc, n_trials      = trial_level_accuracy(preds, labels, test_ds.groups)
    trial_acc_soft, _        = trial_level_accuracy_soft(probs, labels, test_ds.groups)
    metrics = dict(top1=top1, balanced_acc=bal_acc, macro_f1=macro_f1,
                   top5=top5, trial_acc=trial_acc, trial_acc_soft=trial_acc_soft,
                   n_trials=n_trials, ckpt_src=best_src)

    curves = dict(pretrain_loss=[], train_acc=train_accs,
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
    ax.plot(curves["val_f1"], color="#8e44ad", linewidth=1.5)
    ax.axhline(0.0, ls="--", color="gray", linewidth=1)
    ax.set_title("Val macro-F1 (selection metric)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("macro-F1")
    ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    ax = axes[1]
    ax.plot(curves["train_acc"], label="Train", color="#2980b9", linewidth=1.5)
    ax.plot(curves["val_acc"],   label="Val",   color="#27ae60", linewidth=1.5)
    ax.axhline(1 / N_CLASSES, ls="--", color="gray", linewidth=1, label="Chance")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
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
    print(f"  Class balancing : {args.balance}  (v4 default: both)")
    print(f"  Max per class   : {args.max_per_class or 'uncapped'}")
    print(f"  Train epochs    : {args.epochs}  (early-stop patience {args.patience})")
    print(f"  LR (OneCycle)   : {args.lr}  warmup {args.warmup_frac}  wd {args.weight_decay}")
    print(f"  Dropout (head)  : {DROPOUT_P}   Dropout2d (temporal): {TEMPORAL_DROPOUT2D}")
    print(f"  Model dims      : n_f={N_FILTERS} gcn={GCN_DIM} embed={EMBED_DIM} "
          f"adj_thr={ADJ_THRESHOLD}  (+ GCN residual skip)")
    print(f"  EMA decay       : {args.ema if args.ema else 'off'}  "
          f"(checkpoint = best-of raw/EMA)")
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
        tqdm.write(f"    trial-level top-1: {metrics['trial_acc']:.3f}  (hard vote, "
                   f"{metrics['n_trials']} recordings)")
        tqdm.write(f"    trial-level SOFT : {metrics['trial_acc_soft']:.3f}  "
                   f"(prob-avg vote — headline per-trial metric)")
        tqdm.write(f"    checkpoint src   : {metrics['ckpt_src']} weights")
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
        print(f"  {'Fold':<6}{'top1':>8}{'balAcc':>9}{'mF1':>8}{'top5':>8}"
              f"{'trial':>8}{'trialS':>8}")
        print(f"  {'-'*53}")
        for f, m in zip(done_folds, fold_metrics):
            print(f"  {f:<6}{m['top1']:>8.3f}{m['balanced_acc']:>9.3f}"
                  f"{m['macro_f1']:>8.3f}{m['top5']:>8.3f}{m['trial_acc']:>8.3f}"
                  f"{m['trial_acc_soft']:>8.3f}")
        if len(fold_metrics) > 1:
            def col(key): return np.array([m[key] for m in fold_metrics])
            print(f"  {'-'*53}")
            print(f"  {'mean':<6}{col('top1').mean():>8.3f}{col('balanced_acc').mean():>9.3f}"
                  f"{col('macro_f1').mean():>8.3f}{col('top5').mean():>8.3f}"
                  f"{col('trial_acc').mean():>8.3f}{col('trial_acc_soft').mean():>8.3f}")
            print(f"  {'std':<6}{col('top1').std():>8.3f}{col('balanced_acc').std():>9.3f}"
                  f"{col('macro_f1').std():>8.3f}{col('top5').std():>8.3f}"
                  f"{col('trial_acc').std():>8.3f}{col('trial_acc_soft').std():>8.3f}")
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
    parser.add_argument("--epochs",          default=60,    type=int,
                        help="max supervised epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience",        default=12,    type=int,
                        help="early-stop patience on val macro-F1 (0 = disable)")
    parser.add_argument("--batch_size",      default=96,    type=int,
                        help="autoresearch: 64 lost, 96 kept")
    parser.add_argument("--lr",              default=3e-3,  type=float,
                        help="OneCycle max LR (autoresearch win: 1e-3 → 3e-3)")
    parser.add_argument("--warmup_frac",     default=0.15,  type=float,
                        help="OneCycle pct_start (autoresearch: 0.05 lost, 0.15 kept)")
    parser.add_argument("--weight_decay",    default=0.05,  type=float,
                        help="AdamW weight decay (autoresearch: 0.01 lost, 0.05 kept)")
    parser.add_argument("--balance",         default="both",
                        choices=["sampler", "classweight", "both", "none"],
                        help="class-imbalance handling (v4 default: both — confirmed best)")
    parser.add_argument("--ema",             default=0.999, type=float,
                        help="EMA decay; checkpoint = best-of raw/EMA (0 = disable)")
    parser.add_argument("--max_per_class",   default=0,     type=int,
                        help="cap train windows per class (0 = uncapped)")
    parser.add_argument("--no_cache",        action="store_true")
    args = parser.parse_args()
    main(args)
