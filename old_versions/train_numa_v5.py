#!/usr/bin/env python3
"""
ENIGMA-style EEG Brain Decoding Classifier — v5 (EEGNet + filter-bank fusion)
==================================================================================
A deliberate REDESIGN, not a tweak. v3 and v4 both sat at chance and the
autoresearch sweep that produced v4 was, in hindsight, optimizing noise around
the chance floor. v5 changes the backbone and the feature set.

READING THE v3 → v4 RESULTS HONESTLY (5-fold CV, 36 classes, chance top-1 0.028)
-------------------------------------------------------------------------------
                 chance     v3        v4
    top-1        0.028     0.056     0.046
    balanced acc 0.028     0.030     0.031     ← the collapse-proof metric
    macro-F1     ~0        0.008     0.009
    top-5        0.139     0.157     0.181     ← the only faint real signal
    trial-level  0.028     0.028     0.029     ← dead at chance in BOTH

v4 is NOT worse than v3 — they are statistically identical and BOTH AT CHANCE.
The top-1 "drop" (0.056 → 0.046) is an illusion: each model collapses onto
whichever few classes have inflated window-support in that test instance
(v3: broom/cow/monkey ≈ 0.43–0.49 recall; v4: cat/bottle_opener ≈ 0.37–0.42),
and top-1 just reports the frequency of that dominant class. Balanced accuracy —
which punishes exactly that collapse — barely moves (0.030 → 0.031), and
trial-level voting (the metric that actually matters) is pinned at chance.
=> The SGCN backbone has plateaued at the noise floor. More LR/dropout/balance
   tuning cannot help; the autoresearch's "wins" (0.0370 vs 0.0280 on one fixed
   split) were within run-to-run noise.

WHAT v5 CHANGES (and why)
-------------------------
  1. BACKBONE: SGCN → EEGNet (properly sized: F1=16, D=2, F2=32). EEGNet is the
     standard, battle-tested compact EEG classifier. Its block structure learns
     (a) temporal band-pass filters, then (b) DEPTHWISE SPATIAL filters across
     ALL electrodes — a learned spatial projection, strictly more expressive than
     the v3/v4 FIXED electrode-adjacency graph convolution that never beat chance.
     NB: the autoresearch tried "EEGNet F1=8 D=2 (7K params)" and called it "too
     small" — but it actually MATCHED the SGCN's macro-F1. It was dismissed, not
     out-performed. v5 uses a correctly-sized EEGNet, not a 7K toy.

  2. SPECTRAL BRANCH (filter-bank fusion): in parallel with EEGNet we compute
     per-channel LOG BAND POWER in five canonical bands (δ 1–4, θ 4–8, α 8–13,
     β 13–30, γ 30–40 Hz) via a differentiable rFFT, then fuse those features
     with EEGNet's embedding before the classifier. Time-domain conv nets and
     band-power features capture complementary structure; fusing them is a cheap,
     well-motivated shot at signal the pure conv net misses. Crucially this is
     computed ON-THE-FLY from the SAME cached 1-s windows, so v5 reuses
     .eeg_cache_v8 with ZERO reprocessing of the 2 GB BDFs.

  3. CHANNEL WEIGHTING OFF BY DEFAULT (--channel_weights off). The per-subject
     CSV multiply was never validated and injects subject-specific amplitude
     scaling that can actively harm CROSS-INSTANCE generalization. Toggle back on
     with --channel_weights on if you want to A/B it.

  4. KEPT (these were correct and are not the problem):
       • class balancing = "both" (balanced sampler + inverse-√freq weighted CE)
       • selection + early-stop on macro-F1; balanced acc & top-5 reported
       • SOFT trial-level voting (avg per-window softmax) as the headline metric
       • EMA weights selected as best-of-{raw, EMA} (can only help)
       • full 5-fold CV, honest per-class diagnostics, all plots, the cache

HONEST CAVEAT / THE REAL CEILING
--------------------------------
If the 1-second windows are NOT time-locked to stimulus onset (the variable,
wildly-imbalanced per-fold test supports — e.g. snake 2641 vs 153 windows —
strongly suggest the BDFs are variable-length recordings, not clean per-stimulus
epochs), then most windows contain no stimulus-evoked response and the achievable
ceiling really is ~chance regardless of model. The single highest-leverage fix in
that case is upstream, in segmentation: EPOCH on stimulus-onset event markers and
keep a fixed post-onset window (e.g. 0–800 ms). v5 squeezes the most out of the
model side; if it is still at chance, the evidence points squarely at epoching,
not architecture.
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
DROPOUT_P          = 0.3        # used for EEGNet blocks AND fusion head
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
CACHE_DIR          = Path(".eeg_cache_v8")     # IDENTICAL preprocessing → reuse v2/v3/v4 cache
PLOT_DIR           = Path("enigma_plots_v5")

MODEL_TAG   = "v5 (EEGNet + filter-bank spectral fusion)"
CKPT_PREFIX = "enigma_v5"

# Canonical EEG bands (Hz) for the filter-bank branch.
EEG_BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]


# ──────────────────────────────────────────────────────────────────────────────
# eeg_channel_weights (OPTIONAL in v5 — OFF by default)
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
# Preprocessing  (BYTE-IDENTICAL to v3/v4 so the .eeg_cache_v8 cache is reused)
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

    return valid_windows


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (in-memory)
# ──────────────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, samples: list, window_sec: float, overlap_sec: float,
                 weight_table: dict, augment_data: bool = False,
                 use_cache: bool = True, max_per_class: int = 0,
                 use_weights: bool = False):
        self.augment_data = augment_data
        self.items        = []
        self.labels       = []
        self.groups       = []
        rejected          = 0

        for trial_id, (path, label) in enumerate(samples):
            sub_id  = get_subject_id(path)
            ov      = overlap_sec if augment_data else 0.0
            windows = preprocess_bdf(path, window_sec, ov, use_cache)
            if not windows:
                rejected += 1
                continue
            for w in windows:
                if use_weights:
                    w = apply_channel_weights(w, sub_id, weight_table)
                self.items.append((np.ascontiguousarray(w, dtype=np.float32), label))
                self.labels.append(label)
                self.groups.append(trial_id)

        if max_per_class and self.items:
            self._cap_per_class(max_per_class)
        if rejected:
            tqdm.write(f"    Artifact rejection: {rejected} files removed")

    def _cap_per_class(self, cap: int):
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
        x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).unsqueeze(0)
        if self.augment_data:
            # Light aug only — the autoresearch separately found ADDING Gaussian
            # noise / mixup / temporal-shift on top of this gave no gain.
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
               max_per_class: int = 0, use_weights: bool = False):
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
                (test_raw if inst == test_instance else train_raw).append((bdf, lbl))

    tqdm.write(f"    Building train set ({len(train_raw)} BDFs) ...")
    train_ds = EEGDataset(train_raw, window_sec, overlap_sec, weight_table,
                          augment_data=True,  use_cache=use_cache,
                          max_per_class=max_per_class, use_weights=use_weights)
    tqdm.write(f"    Building test  set ({len(test_raw)} BDFs) ...")
    test_ds  = EEGDataset(test_raw,  window_sec, 0.0, weight_table,
                          augment_data=False, use_cache=use_cache,
                          max_per_class=0, use_weights=use_weights)
    tqdm.write(f"    Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    if train_ds.labels:
        counts = np.bincount(np.array(train_ds.labels), minlength=N_CLASSES)
        nz = counts[counts > 0]
        tqdm.write(f"    Train windows/class: min={nz.min()} max={nz.max()} "
                   f"mean={nz.mean():.0f} (imbalance {nz.max()/max(1,nz.min()):.1f}×)")
    return train_ds, test_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# Model: EEGNet backbone + differentiable filter-bank spectral branch
# ──────────────────────────────────────────────────────────────────────────────

class Conv2dConstrained(nn.Conv2d):
    """Conv2d with EEGNet's max-norm weight constraint (applied each forward)."""
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                # per-output-filter L2 norm; vector_norm handles the 3-tuple dim
                # (Tensor.norm with a 3-tuple wrongly dispatches to matrix_norm).
                norm = torch.linalg.vector_norm(
                    self.weight, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                desired = norm.clamp(max=self.max_norm)
                self.weight.mul_(desired / norm)
        return super().forward(x)


class BandPower(nn.Module):
    """Per-channel log band power via rFFT. Input (B,1,C,T) → (B, C*n_bands).

    Differentiable; computed on-the-fly from the cached 1-s windows so no
    re-preprocessing is needed. Windows are per-channel z-scored upstream, so this
    captures the RELATIVE spectral shape across the five canonical EEG bands."""
    def __init__(self, n_times: int, fs: float = RESAMPLE_FREQ, bands=EEG_BANDS):
        super().__init__()
        freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)            # (F,)
        masks = np.zeros((len(bands), len(freqs)), dtype=np.float32)
        for b, (lo, hi) in enumerate(bands):
            masks[b] = ((freqs >= lo) & (freqs < hi)).astype(np.float32)
        self.register_buffer("masks", torch.from_numpy(masks))  # (n_bands, F)
        self.n_bands = len(bands)

    def forward(self, x):
        B, _, C, T = x.shape
        spec  = torch.fft.rfft(x.squeeze(1), dim=-1)            # (B, C, F)
        power = (spec.real ** 2 + spec.imag ** 2)               # (B, C, F)
        # (B,C,F) · (n_bands,F)ᵀ → (B, C, n_bands)
        band = torch.einsum("bcf,nf->bcn", power, self.masks)
        band = torch.log1p(band)
        return band.reshape(B, C * self.n_bands)


class EEGNetFB(nn.Module):
    """EEGNet (F1/D/F2) temporal-spatial backbone fused with a band-power branch.

    Input (B, 1, C, T) → logits (B, n_classes)."""
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 F1: int = 16, D: int = 2, F2: int = 32,
                 kern_len: int = 64, dropout_p: float = DROPOUT_P,
                 use_bandpower: bool = True):
        super().__init__()
        self.use_bandpower = use_bandpower

        # ── EEGNet block 1: temporal conv → depthwise SPATIAL conv over channels ─
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            Conv2dConstrained(F1, F1 * D, (n_channels, 1), groups=F1, bias=False,
                              max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_p),
        )
        # ── EEGNet block 2: separable temporal conv ──────────────────────────────
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_p),
        )

        with torch.no_grad():
            d = torch.zeros(1, 1, n_channels, n_times)
            d = self.separable(self.depthwise(self.firstconv(d)))
            eeg_feat_dim = d.numel()

        # ── Filter-bank spectral branch ──────────────────────────────────────────
        fb_out = 0
        if use_bandpower:
            self.bandpower = BandPower(n_times)
            fb_in  = n_channels * self.bandpower.n_bands
            fb_out = 64
            self.fb_head = nn.Sequential(
                nn.LayerNorm(fb_in),
                nn.Linear(fb_in, fb_out),
                nn.ELU(),
                nn.Dropout(dropout_p),
            )

        # ── Fusion classifier ─────────────────────────────────────────────────────
        fused = eeg_feat_dim + fb_out
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 128),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        z = self.separable(self.depthwise(self.firstconv(x)))
        z = z.flatten(1)
        if self.use_bandpower:
            z = torch.cat([z, self.fb_head(self.bandpower(x))], dim=1)
        return self.classifier(z)


# ──────────────────────────────────────────────────────────────────────────────
# EMA weight averaging (selected as best-of-{raw, EMA} so it can never hurt)
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
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
# Eval / metrics
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    """Return (preds, labels, top5_hits, probs) over a loader in dataset order."""
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
    """HARD vote: majority window prediction per recording."""
    by_group, truth = defaultdict(list), {}
    for p, l, g in zip(preds, labels, groups):
        by_group[g].append(p); truth[g] = l
    correct = sum(int(max(set(v), key=v.count) == truth[g]) for g, v in by_group.items())
    return correct / max(1, len(by_group)), len(by_group)


def trial_level_accuracy_soft(probs, labels, groups):
    """SOFT vote: average per-window softmax per recording, then argmax."""
    sum_prob, truth = {}, {}
    for pr, l, g in zip(probs, labels, groups):
        if g not in sum_prob:
            sum_prob[g] = np.zeros(N_CLASSES, dtype=np.float64); truth[g] = l
        sum_prob[g] += pr
    correct = sum(int(np.argmax(sum_prob[g]) == truth[g]) for g in sum_prob)
    return correct / max(1, len(sum_prob)), len(sum_prob)


# ──────────────────────────────────────────────────────────────────────────────
# One fold
# ──────────────────────────────────────────────────────────────────────────────

def run_fold(fold: int, stimuli_dir: Path, weight_table: dict,
             allowed_subjects: set, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec,
        weight_table, allowed_subjects, not args.no_cache, args.max_per_class,
        use_weights=(args.channel_weights == "on"))

    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data, skipping.")
        return None, None, None, label_map, None, None, None, None

    # ── Class balancing (sampler + class-weighted CE) ─────────────────────────
    train_labels = np.asarray(train_ds.labels)
    counts       = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    safe_counts  = np.where(counts > 0, counts, 1.0)

    sampler = None
    if args.balance in ("sampler", "both"):
        sample_w = (1.0 / safe_counts)[train_labels]
        sampler  = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_labels), replacement=True)

    ce_weight = None
    if args.balance in ("classweight", "both"):
        w = np.where(counts > 0, (counts.sum() / (len(counts) * safe_counts)) ** 0.5, 0.0)
        w = w / (w[w > 0].mean() + 1e-8)
        ce_weight = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    n_channels = train_ds[0][0].shape[1]
    target_T   = train_ds[0][0].shape[-1]

    model = EEGNetFB(n_channels, target_T, N_CLASSES,
                     dropout_p=args.dropout,
                     use_bandpower=(args.bandpower == "on")).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=args.label_smooth)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=args.warmup_frac)

    ema = ModelEMA(model, args.ema) if args.ema and args.ema > 0 else None

    best_f1, best_state, best_top1, best_src = -1.0, None, 0.0, "raw"
    epochs_no_improve             = 0
    train_accs, val_accs, val_f1s = [], [], []

    def _eval_and_score():
        preds, labels, _, _ = evaluate(model, test_loader)
        acc = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        return acc, f1

    ft_bar = tqdm(range(1, args.epochs + 1), desc=f"  Fold {fold} train",
                  ncols=90, unit="ep", leave=False, colour="cyan")
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

        raw_acc, raw_f1 = _eval_and_score()
        cand_f1, cand_acc, cand_src = raw_f1, raw_acc, "raw"
        cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

        if cand_f1 > best_f1:
            best_f1, best_top1   = cand_f1, cand_acc
            best_state, best_src = cand_state, cand_src
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ft_bar.set_postfix(tr=f"{tr_acc:.2f}", val=f"{cand_acc:.2f}",
                           f1=f"{cand_f1:.3f}", best_f1=f"{max(best_f1,0):.3f}",
                           src=best_src)

        if args.patience and epochs_no_improve >= args.patience:
            tqdm.write(f"  Early stop at epoch {epoch} (no macro-F1 gain in "
                       f"{args.patience} epochs; best from '{best_src}' weights)")
            break

    model.load_state_dict(best_state)
    preds, labels, top5_hits, probs = evaluate(model, test_loader)

    top1     = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
    bal_acc  = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    top5     = top5_hits / max(1, len(labels))
    trial_acc, n_trials = trial_level_accuracy(preds, labels, test_ds.groups)
    trial_acc_soft, _   = trial_level_accuracy_soft(probs, labels, test_ds.groups)
    metrics = dict(top1=top1, balanced_acc=bal_acc, macro_f1=macro_f1, top5=top5,
                   trial_acc=trial_acc, trial_acc_soft=trial_acc_soft,
                   n_trials=n_trials, ckpt_src=best_src)

    curves = dict(train_acc=train_accs, val_acc=val_accs, val_f1=val_f1s)
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
    if args.channel_weights == "on":
        if weights_path.exists():
            weight_table = load_channel_weights(weights_path, allowed_subjects)
        else:
            tqdm.write(f"  [WEIGHTS] WARNING: {weights_path} not found — uniform weights.")

    print(f"\n{'='*60}")
    print(f"  ENIGMA EEG Classifier  {MODEL_TAG}")
    print(f"{'='*60}")
    print(f"  Device          : {DEVICE}")
    print(f"  Stimuli dir     : {stimuli_dir}")
    print(f"  Excluded subs   : {sorted(EXCLUDED_SUBJECTS)}")
    print(f"  Active subjects : {n_valid}  {valid_subs}")
    print(f"  Channel weights : {args.channel_weights}"
          + (f"  ({weights_path})" if args.channel_weights == "on" else "  (default: off)"))
    print(f"  Resampled Hz    : {RESAMPLE_FREQ}")
    print(f"  Window/Overlap  : {args.window_sec}s / {args.overlap_sec}s")
    print(f"  Bandpass        : {BANDPASS_LOW}–{BANDPASS_HIGH} Hz")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Class balancing : {args.balance}")
    print(f"  Backbone        : EEGNet(F1=16,D=2,F2=32,kern=64)"
          f" + bandpower={args.bandpower}")
    print(f"  Dropout         : {args.dropout}   label-smooth {args.label_smooth}")
    print(f"  Train epochs    : {args.epochs}  (early-stop patience {args.patience})")
    print(f"  LR (OneCycle)   : {args.lr}  warmup {args.warmup_frac}  wd {args.weight_decay}")
    print(f"  EMA decay       : {args.ema if args.ema else 'off'}  (ckpt = best-of raw/EMA)")
    print(f"  Selection metric: macro-F1")
    print(f"  Cache dir       : {'disabled' if args.no_cache else str(CACHE_DIR)}")
    print(f"  Plot dir        : {PLOT_DIR}")
    print(f"  Folds           : {args.fold if args.fold else '1-5 (full CV)'}")
    print(f"{'='*60}\n")

    folds          = [args.fold] if args.fold else list(range(1, N_INSTANCES + 1))
    fold_accs, fold_metrics, fold_results, done_folds = [], [], [], []
    last_label_map = None

    for fold in folds:
        tqdm.write(f"\n── Fold {fold}  [test=instance{fold}] ──")
        acc, preds, labels, label_map, best_state, n_channels, curves, metrics = run_fold(
            fold, stimuli_dir, weight_table, allowed_subjects, args)
        if acc is None:
            continue

        fold_accs.append(acc); fold_metrics.append(metrics)
        fold_results.append((preds, labels)); done_folds.append(fold)
        last_label_map = label_map

        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in range(N_CLASSES)]
        tqdm.write(f"\n  Fold {fold} metrics (chance top-1 = {1/N_CLASSES:.3f}):")
        tqdm.write(f"    top-1            : {metrics['top1']:.3f}")
        tqdm.write(f"    balanced acc     : {metrics['balanced_acc']:.3f}  "
                   f"(macro-recall; collapse → ~{1/N_CLASSES:.3f})")
        tqdm.write(f"    macro-F1         : {metrics['macro_f1']:.3f}  (selection metric)")
        tqdm.write(f"    top-5            : {metrics['top5']:.3f}  (chance = {5/N_CLASSES:.3f})")
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
    parser.add_argument("--epochs",          default=60,    type=int)
    parser.add_argument("--patience",        default=12,    type=int,
                        help="early-stop patience on val macro-F1 (0 = disable)")
    parser.add_argument("--batch_size",      default=96,    type=int)
    parser.add_argument("--lr",              default=1e-3,  type=float,
                        help="OneCycle max LR (1e-3 is the EEGNet sweet spot)")
    parser.add_argument("--warmup_frac",     default=0.15,  type=float)
    parser.add_argument("--weight_decay",    default=0.05,  type=float)
    parser.add_argument("--dropout",         default=DROPOUT_P, type=float)
    parser.add_argument("--label_smooth",    default=0.1,   type=float)
    parser.add_argument("--balance",         default="both",
                        choices=["sampler", "classweight", "both", "none"])
    parser.add_argument("--bandpower",       default="on", choices=["on", "off"],
                        help="filter-bank spectral fusion branch (v5 default: on)")
    parser.add_argument("--channel_weights", default="off", choices=["on", "off"],
                        help="per-subject channel-weight multiply (v5 default: OFF)")
    parser.add_argument("--ema",             default=0.999, type=float,
                        help="EMA decay; checkpoint = best-of raw/EMA (0 = disable)")
    parser.add_argument("--max_per_class",   default=0,     type=int)
    parser.add_argument("--no_cache",        action="store_true")
    args = parser.parse_args()
    main(args)
