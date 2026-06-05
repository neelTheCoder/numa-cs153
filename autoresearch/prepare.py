#!/usr/bin/env python3
"""
prepare.py — FIXED autoresearch infrastructure for the ENIGMA EEG classifier.
================================================================================
This file is the **read-only ground truth** of the autoresearch loop, modeled on
Karpathy's `autoresearch` (miolini/autoresearch-macos). The research agent
editing `train.py` is NOT allowed to modify this file. It owns everything that
defines *what success means* and *what data is seen*, so that every experiment is
measured against an identical, un-gameable benchmark:

  * fixed preprocessing of the raw BDF recordings -> cached float32 windows
  * a fixed train/val split (leave-one-instance-out, instance 5 = validation)
  * the fixed evaluation harness `evaluate()` and the scoring function `score()`
  * fixed task constants (classes, sampling rate, window length, time budget)

Run this ONCE up front to build/verify the window cache:

    python prepare.py                # build cache (reuses existing if present)
    python prepare.py --check        # just report data/cache status

`train.py` imports `build_split`, `evaluate`, `score`, and the constants below.

Supported dataset layouts (auto-detected):
  multi-subject : <root>/<class>/sub-NN/instance{1..5}.bdf   (remote GPU box)
  single-subject: <root>/<class>/instance{1..5}.bdf          (local mac)
================================================================================
DO NOT MODIFY THIS FILE during autoresearch. It is the evaluation contract.
================================================================================
"""

import argparse
import hashlib
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

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
    from torch.utils.data import Dataset
except ImportError:
    sys.exit("pip install torch")

try:
    from sklearn.metrics import balanced_accuracy_score, f1_score
except ImportError:
    sys.exit("pip install scikit-learn")

import functools
print = functools.partial(print, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# FIXED TASK CONSTANTS  (the evaluation contract — DO NOT CHANGE)
# ──────────────────────────────────────────────────────────────────────────────
N_CLASSES         = 36          # number of object stimuli (classes)
N_INSTANCES       = 5           # repetitions per class
VAL_INSTANCE      = 5           # instance held out for validation (fixed)
EXCLUDED_SUBJECTS = {5, 8, 12}  # subjects dropped (multi-subject layout only)
ALL_SUBJECTS      = set(range(1, 13))

RESAMPLE_FREQ     = 256.0       # Hz
BANDPASS_LOW      = 1.0         # Hz
BANDPASS_HIGH     = 40.0        # Hz
NOTCH_FREQ        = 50.0        # Hz
ARTIFACT_THRESH   = 150e-6      # peak-to-peak artifact rejection (volts)
WINDOW_SEC        = 1.0         # window length seconds (fixed eval contract)
TRAIN_OVERLAP     = 0.5         # train-window overlap seconds
CHANCE            = 1.0 / N_CLASSES

# Cap on TRAIN windows per class (bounds RAM + load time so each experiment is
# fast and comparable). Val is uncapped. 0 = uncapped. Part of the fixed
# contract — identical for every experiment.
MAX_TRAIN_PER_CLASS = 1200

# Per-experiment wall-clock TRAINING budget, seconds. The agent trains until this
# elapses, then evaluates. Kept comfortably under 10 min so the whole experiment
# (data load + train + eval) stays within the program.md 10-minute hard cap.
TIME_BUDGET       = 300         # 5 minutes of training (≈7 min/experiment incl. data load + eval, under the 10-min cap)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ──────────────────────────────────────────────────────────────────────────────
# Locate the stimuli root + cache dir (robust across the local mac & remote box)
# ──────────────────────────────────────────────────────────────────────────────
def _find_stimuli_root() -> Path:
    env = os.environ.get("AUTORESEARCH_STIMULI")
    candidates = [env] if env else []
    candidates += [
        "stimuli", "../stimuli",
        os.path.expanduser("~/cleaned_data/cleaned_data"),
        "cleaned_data/cleaned_data", "../cleaned_data/cleaned_data",
        os.path.expanduser("~/cleaned_data"),
    ]
    for c in candidates:
        if c and Path(c).is_dir() and any(Path(c).iterdir()):
            return Path(c)
    sys.exit("Could not locate stimuli root. Set AUTORESEARCH_STIMULI=/path/to/data")


def _find_cache_dir() -> Path:
    env = os.environ.get("AUTORESEARCH_CACHE")
    if env:
        return Path(env)
    # Reuse the existing 6GB cache on the remote box if present (same hash scheme).
    existing = Path(os.path.expanduser("~/.eeg_cache_v8"))
    if existing.is_dir():
        return existing
    return Path(".eeg_cache_autoresearch")


STIMULI_DIR = _find_stimuli_root()
CACHE_DIR   = _find_cache_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Fixed preprocessing: raw BDF -> list of (C, T) float32 windows
# (Hash scheme matches train_enigma_v3.py so the existing cache is reused.)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_bdf(path: Path, overlap_sec: float) -> list:
    """Deterministic preprocessing with on-disk float32 cache. DO NOT CHANGE."""
    key = hashlib.md5(
        f"{path}{WINDOW_SEC}{overlap_sec}{RESAMPLE_FREQ}"
        f"{BANDPASS_LOW}{BANDPASS_HIGH}".encode()
    ).hexdigest()
    cache_path = CACHE_DIR / f"{key}.npy"

    if cache_path.exists():
        try:
            arr = np.load(str(cache_path), allow_pickle=False)  # (N, C, T) f32
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
    target_T = int(RESAMPLE_FREQ * WINDOW_SEC)
    step_T   = max(1, int(RESAMPLE_FREQ * (WINDOW_SEC - overlap_sec)))

    windows = []
    for start in range(0, data.shape[1] - target_T + 1, step_T):
        w = data[:, start: start + target_T].copy()
        if np.ptp(w, axis=1).max() > ARTIFACT_THRESH:
            continue
        bs = max(1, int(target_T * 0.2))
        w -= w[:, :bs].mean(axis=1, keepdims=True)
        w  = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
        windows.append(w.astype(np.float32))

    if windows:
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(str(cache_path), np.stack(windows).astype(np.float32))
    return windows


# ──────────────────────────────────────────────────────────────────────────────
# Fixed dataset
# ──────────────────────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    """In-memory windows. `augment` adds light noise/scaling (train only).
    `cap` limits windows kept per class (0 = uncapped)."""
    def __init__(self, samples, overlap_sec, augment=False, cap=0):
        self.augment = augment
        self.items, self.labels, self.groups = [], [], []
        per_class = defaultdict(int)
        for trial_id, (path, label) in enumerate(samples):
            if cap and per_class[label] >= cap:
                continue
            for w in preprocess_bdf(Path(path), overlap_sec):
                if cap and per_class[label] >= cap:
                    break
                self.items.append((np.ascontiguousarray(w, dtype=np.float32), label))
                self.labels.append(label)
                self.groups.append(trial_id)
                per_class[label] += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data, label = self.items[idx]
        x = torch.from_numpy(np.array(data, dtype=np.float32, copy=True)).unsqueeze(0)
        if self.augment:
            x = x + torch.randn_like(x) * 0.05
            mask = (torch.rand(x.shape[1], 1) > 0.10).float()
            x = x * mask.unsqueeze(0)
            x = x * (0.9 + 0.2 * torch.rand(1).item())
        return x, label


def _discover_samples():
    """Return (label_map, by_instance) handling both dataset layouts."""
    classes = sorted(d.name for d in STIMULI_DIR.iterdir() if d.is_dir())
    label_map = {cls: i for i, cls in enumerate(classes)}
    by_instance = defaultdict(list)  # instance -> [(path, label), ...]
    allowed = ALL_SUBJECTS - EXCLUDED_SUBJECTS
    for cls in classes:
        lbl = label_map[cls]
        cls_dir = STIMULI_DIR / cls
        sub_dirs = [d for d in sorted(cls_dir.iterdir()) if d.is_dir()
                    and re.match(r"sub-?\d+", d.name)]
        if sub_dirs:  # multi-subject layout
            for sub_dir in sub_dirs:
                m = re.match(r"sub-?(\d+)", sub_dir.name)
                if not m or int(m.group(1)) not in allowed:
                    continue
                for inst in range(1, N_INSTANCES + 1):
                    bdf = sub_dir / f"instance{inst}.bdf"
                    if bdf.exists():
                        by_instance[inst].append((bdf, lbl))
        else:         # single-subject layout
            for inst in range(1, N_INSTANCES + 1):
                bdf = cls_dir / f"instance{inst}.bdf"
                if bdf.exists():
                    by_instance[inst].append((bdf, lbl))
    return label_map, by_instance


def build_split():
    """
    Build the FIXED train/val split (leave-instance-`VAL_INSTANCE`-out).
    Returns (train_ds, val_ds, label_map). Identical across every experiment.
    """
    label_map, by_instance = _discover_samples()
    train_raw, val_raw = [], []
    for inst, items in by_instance.items():
        (val_raw if inst == VAL_INSTANCE else train_raw).extend(items)

    # Deterministic shuffle so the per-class cap draws a subject-diverse subset.
    rng = np.random.default_rng(0)
    rng.shuffle(train_raw)

    train_ds = EEGDataset(train_raw, TRAIN_OVERLAP, augment=True,
                          cap=MAX_TRAIN_PER_CLASS)
    val_ds   = EEGDataset(val_raw,   0.0,           augment=False, cap=0)
    return train_ds, val_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# FIXED evaluation harness + scoring  (the ground-truth metric — DO NOT CHANGE)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    """Run `model` over `loader`; return (preds, labels, top5_hits)."""
    model.eval()
    preds, labels, top5_hits = [], [], 0
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        k = min(5, logits.shape[1])
        top5 = logits.topk(k, dim=1).indices.cpu()
        ys = yb.tolist()
        for i, y in enumerate(ys):
            top5_hits += int(y in top5[i].tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(ys)
    return preds, labels, top5_hits


def score(preds, labels, top5_hits, groups=None):
    """
    Fixed metric bundle. PRIMARY target = `balanced_acc` (macro-averaged recall
    over 36 classes) = honest classification accuracy on this imbalanced task.
    Top-1 is reported but NOT the target (inflatable by single-class collapse).
    """
    n = max(1, len(labels))
    top1 = sum(p == l for p, l in zip(preds, labels)) / n
    bal  = balanced_accuracy_score(labels, preds) if labels else 0.0
    mf1  = f1_score(labels, preds, average="macro", zero_division=0) if labels else 0.0
    top5 = top5_hits / n

    trial_acc = 0.0
    if groups is not None:
        by_group, truth = defaultdict(list), {}
        for p, l, g in zip(preds, labels, groups):
            by_group[g].append(p); truth[g] = l
        if by_group:
            correct = sum(int(max(set(v), key=v.count) == truth[g])
                          for g, v in by_group.items())
            trial_acc = correct / len(by_group)

    return dict(balanced_acc=bal, top1=top1, macro_f1=mf1,
                top5=top5, trial_acc=trial_acc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: build cache / status check
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Build/verify the autoresearch EEG cache")
    ap.add_argument("--check", action="store_true",
                    help="only report data/cache status; do not build")
    args = ap.parse_args()

    label_map, by_instance = _discover_samples()
    n_bdf = sum(len(v) for v in by_instance.values())
    print("=" * 64)
    print("  autoresearch prepare.py — ENIGMA EEG classification")
    print("=" * 64)
    print(f"  device          : {DEVICE}")
    print(f"  stimuli root    : {STIMULI_DIR.resolve()}")
    print(f"  cache dir       : {CACHE_DIR.resolve()}")
    print(f"  classes         : {len(label_map)} (expected {N_CLASSES})")
    print(f"  BDF recordings  : {n_bdf}")
    print(f"  excluded subs   : {sorted(EXCLUDED_SUBJECTS)}")
    print(f"  val instance    : {VAL_INSTANCE} (held out)")
    print(f"  window / overlap: {WINDOW_SEC}s / {TRAIN_OVERLAP}s (train)")
    print(f"  cap/class train : {MAX_TRAIN_PER_CLASS or 'uncapped'}")
    print(f"  time budget     : {TIME_BUDGET}s per experiment")
    print(f"  chance balanced : {CHANCE:.4f}")
    print("=" * 64)

    if args.check:
        n_cached = len(list(CACHE_DIR.glob("*.npy"))) if CACHE_DIR.exists() else 0
        print(f"  cached window files: {n_cached}")
        print("  status:", "cache present" if n_cached else
              "no cache yet — run `python prepare.py` (will preprocess; slow once)")
        return

    print("  Building/loading window cache (one-time preprocess; then instant)...")
    t0 = time.time()
    train_ds, val_ds, _ = build_split()
    dt = time.time() - t0
    tr_counts = np.bincount(np.asarray(train_ds.labels), minlength=N_CLASSES)
    nz = tr_counts[tr_counts > 0]
    print(f"  train windows: {len(train_ds)} | val windows: {len(val_ds)}")
    if len(nz):
        print(f"  train windows/class: min={nz.min()} max={nz.max()} "
              f"mean={nz.mean():.0f}")
    print(f"  build took {dt:.1f}s")
    print("  Done. Now run a baseline:  python train.py")


if __name__ == "__main__":
    main()
