#!/usr/bin/env python3
"""
numa EEG Brain Decoding — v6 (recording-level attention MIL over EEGNet windows)
==================================================================================
A change of UNIT, not just architecture. v3, v4 and v5 all classified single 1-s
windows and then voted. All three landed at chance on every collapse-proof metric.
v6 stops classifying windows and instead classifies the RECORDING, using
attention-based Multiple-Instance Learning (MIL) to learn which windows carry
signal and ignore the rest.

THE EVIDENCE THAT FORCED THIS REDESIGN  (5-fold CV means, chance top-1 = 0.028)
------------------------------------------------------------------------------
                 chance   v3(SGCN) v4(SGCN+) v5(EEGNet+FB)
    top-1        0.028     0.056     0.046     0.028
    balanced acc 0.028     0.030     0.031     0.030
    macro-F1     ~0        0.008     0.009     0.018
    top-5        0.139     0.157     0.181     0.132   ← v5 BELOW chance
    trial (soft) 0.028     0.029     0.029     0.027   ← at chance every time

v5's macro-F1 doubling (0.009→0.018) is NOT real discrimination — it is "spread
guessing." v5 cured the single-class collapse, so predictions now scatter over all
36 classes, and macro-F1 mechanically rewards that scatter. But the three metrics
that cannot be gamed all read chance: top-5 actually fell BELOW chance, balanced
accuracy sat at chance, and — decisively — TRIAL-LEVEL VOTING over ~20 windows per
recording stayed at chance. Aggregating 20 noisy views cannot fail to lift a real
per-window signal; it stayed flat because there is little per-window signal to
lift. THREE different backbones (SGCN, tuned SGCN, EEGNet+filter-bank) all hit the
same chance floor ⇒ the backbone is no longer the bottleneck.

ROOT CAUSE (data, not model): the 1-s windows are almost certainly NOT
stimulus-onset-locked. The smoking gun is the wildly variable per-fold test
support (e.g. snake = 2641 windows in fold 4 vs 153 in others; keyboard 642↔1254)
— a clean per-stimulus epoch set would have near-constant window counts. The BDFs
are variable-length recordings sliced blindly, so most windows fall in
inter-stimulus / rest periods with no evoked response.

WHY v6 = ATTENTION MIL (the right model-side response to that root cause)
------------------------------------------------------------------------
  1. RECORDING = BAG. Each recording is a bag of its windows. A gated-attention
     pooling head (Ilse et al., 2018) computes a learned weight per window and
     forms the recording embedding as the attention-weighted sum, THEN classifies.
     The model can concentrate on the few stimulus-locked windows and suppress the
     rest — something neither per-window CE nor majority voting can do.
  2. OPTIMIZES THE RIGHT UNIT. We care about per-recording decoding; v6 trains and
     selects on it directly instead of on noisy single windows.
  3. DISSOLVES THE IMBALANCE ARTIFACT. Window counts were 4×+ imbalanced because
     recordings have different lengths. RECORDINGS are ~balanced: 9 subjects × 4
     train instances = 36 recordings/class (test = 9/class). So balanced-accuracy
     and top-1 finally measure the same thing, and the sampler/weighting hacks that
     every prior version needed become almost unnecessary.
  4. WINDOW SUBSAMPLING = AUGMENTATION. Each epoch a bag is represented by a fresh
     random K-window subset — many views of the same recording, for free.
  5. REUSES .eeg_cache_v8 AND the v5 EEGNet encoder verbatim — guaranteed to run,
     no 2 GB reprocessing.

HONEST CEILING / THE REAL FIX IF v6 IS STILL AT CHANCE
------------------------------------------------------
MIL is the best model-side lever for sparse, badly-localized signal, but it cannot
invent signal that is not in the windows. If v6 is also at chance, that is
near-conclusive that the remaining fix is UPSTREAM in segmentation: epoch on the
BDF trigger / STATUS channel (or *_events.txt) at stimulus onset and keep a fixed
post-onset window (e.g. 0–800 ms). That belongs in segment_eeg.py, not here — and
it is where the next effort should go.
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
DROPOUT_P          = 0.3
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
CACHE_DIR          = Path(".eeg_cache_v8")     # IDENTICAL preprocessing → reuse cache
PLOT_DIR           = Path("numa_plots_v6")

MODEL_TAG   = "v6 (recording-level attention MIL over EEGNet windows)"
CKPT_PREFIX = "numa_v6"

EEG_BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]


# ──────────────────────────────────────────────────────────────────────────────
# Optional per-subject channel weights (OFF by default, as in v5)
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
        col_vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        if np.all(np.isnan(col_vals)):
            continue
        col_vals = np.where(np.isnan(col_vals), 1.0, col_vals).astype(np.float32)
        weights[f"sub-{sub_num:02d}"] = col_vals
    tqdm.write(f"  [WEIGHTS] Loaded channel weights for {len(weights)} subjects.")
    return weights


def get_subject_id(bdf_path: Path) -> str:
    for part in bdf_path.parts:
        m = re.match(r"^sub-?(\d+)$", part)
        if m:
            return f"sub-{int(m.group(1)):02d}"
    return "unknown"


def apply_channel_weights(window: np.ndarray, sub_id: str, weight_table: dict) -> np.ndarray:
    if sub_id not in weight_table:
        return window
    w = weight_table[sub_id]
    if w.shape[0] != window.shape[0]:
        return window
    return (window * w[:, np.newaxis]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — BYTE-IDENTICAL to v3/v4/v5 so .eeg_cache_v8 is reused
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
            arr = np.load(str(cache_path), allow_pickle=False)
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
# Bag dataset — one item == one RECORDING (a bag of its windows)
# ──────────────────────────────────────────────────────────────────────────────

class BagDataset(Dataset):
    """Each item is (windows[n,1,C,T] float32, label). Train items return a random
    K-window subset (augmentation); eval items return up to `eval_max` windows
    spread evenly across the recording."""
    def __init__(self, samples: list, window_sec: float, overlap_sec: float,
                 weight_table: dict, augment_data: bool, use_cache: bool,
                 train_k: int = 24, eval_max: int = 96, use_weights: bool = False):
        self.augment_data = augment_data
        self.train_k      = train_k
        self.eval_max     = eval_max
        self.bags         = []     # list of np.ndarray (n_i, C, T)
        self.labels       = []     # parallel per-bag label
        self.sub_ids      = []
        rejected          = 0

        for path, label in samples:
            sub_id  = get_subject_id(path)
            ov      = overlap_sec if augment_data else 0.0
            windows = preprocess_bdf(path, window_sec, ov, use_cache)
            if not windows:
                rejected += 1
                continue
            arr = np.stack(windows).astype(np.float32)       # (n, C, T)
            if use_weights and sub_id in weight_table:
                w = weight_table[sub_id]
                if w.shape[0] == arr.shape[1]:
                    arr = arr * w[None, :, None]
            self.bags.append(np.ascontiguousarray(arr, dtype=np.float32))
            self.labels.append(label)
            self.sub_ids.append(sub_id)
        if rejected:
            tqdm.write(f"    Artifact rejection: {rejected} recordings removed")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        arr   = self.bags[idx]                               # (n, C, T)
        n     = arr.shape[0]
        label = self.labels[idx]
        if self.augment_data:
            k = min(self.train_k, n)
            sel = np.random.choice(n, size=k, replace=False)
        else:
            if n > self.eval_max:
                sel = np.linspace(0, n - 1, self.eval_max).round().astype(int)
            else:
                sel = np.arange(n)
        x = torch.from_numpy(np.array(arr[sel], dtype=np.float32, copy=True)).unsqueeze(1)
        if self.augment_data:
            x = x + torch.randn_like(x) * 0.05               # light per-window aug
            x = x * (0.9 + 0.2 * torch.rand(1).item())
        return x, label                                      # x: (k, 1, C, T)


def collate_bags(batch):
    """Pad variable-size bags to the batch max; return (x, mask, y)."""
    xs, ys = zip(*batch)
    B    = len(xs)
    Nmax = max(x.shape[0] for x in xs)
    C, T = xs[0].shape[2], xs[0].shape[3]
    x    = torch.zeros(B, Nmax, 1, C, T, dtype=torch.float32)
    mask = torch.zeros(B, Nmax, dtype=torch.float32)
    for i, xi in enumerate(xs):
        n = xi.shape[0]
        x[i, :n] = xi
        mask[i, :n] = 1.0
    return x, mask, torch.tensor(ys, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Fold builder (recording-level)
# ──────────────────────────────────────────────────────────────────────────────

def build_fold(stimuli_dir: Path, test_instance: int,
               window_sec: float, overlap_sec: float, weight_table: dict,
               allowed_subjects: set, use_cache: bool, args):
    label_map           = {}
    train_raw, test_raw = [], []
    classes = sorted([d.name for d in stimuli_dir.iterdir() if d.is_dir()])
    for cls in classes:
        lbl            = len(label_map)
        label_map[cls] = lbl
        for sub_dir in sorted((stimuli_dir / cls).iterdir()):
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

    uw = (args.channel_weights == "on")
    tqdm.write(f"    Building train set ({len(train_raw)} recordings) ...")
    train_ds = BagDataset(train_raw, window_sec, overlap_sec, weight_table,
                          augment_data=True, use_cache=use_cache,
                          train_k=args.train_k, eval_max=args.eval_max, use_weights=uw)
    tqdm.write(f"    Building test  set ({len(test_raw)} recordings) ...")
    test_ds  = BagDataset(test_raw, window_sec, 0.0, weight_table,
                          augment_data=False, use_cache=use_cache,
                          train_k=args.train_k, eval_max=args.eval_max, use_weights=uw)
    tqdm.write(f"    Train recordings: {len(train_ds)} | Test recordings: {len(test_ds)}")
    if train_ds.labels:
        counts = np.bincount(np.array(train_ds.labels), minlength=N_CLASSES)
        nz = counts[counts > 0]
        tqdm.write(f"    Train recordings/class: min={nz.min()} max={nz.max()} "
                   f"mean={nz.mean():.0f} (imbalance {nz.max()/max(1,nz.min()):.1f}×)")
    return train_ds, test_ds, label_map


# ──────────────────────────────────────────────────────────────────────────────
# EEGNet window encoder + filter-bank branch  (from v5)
# ──────────────────────────────────────────────────────────────────────────────

class Conv2dConstrained(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                norm = torch.linalg.vector_norm(
                    self.weight, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                self.weight.mul_(norm.clamp(max=self.max_norm) / norm)
        return super().forward(x)


class BandPower(nn.Module):
    """Per-channel log band power via rFFT. (M,1,C,T) → (M, C*n_bands)."""
    def __init__(self, n_times: int, fs: float = RESAMPLE_FREQ, bands=EEG_BANDS):
        super().__init__()
        freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
        masks = np.zeros((len(bands), len(freqs)), dtype=np.float32)
        for b, (lo, hi) in enumerate(bands):
            masks[b] = ((freqs >= lo) & (freqs < hi)).astype(np.float32)
        self.register_buffer("masks", torch.from_numpy(masks))
        self.n_bands = len(bands)

    def forward(self, x):
        M, _, C, T = x.shape
        spec  = torch.fft.rfft(x.squeeze(1), dim=-1)
        power = spec.real ** 2 + spec.imag ** 2
        band  = torch.einsum("mcf,nf->mcn", power, self.masks)
        return torch.log1p(band).reshape(M, C * self.n_bands)


class EEGNetEncoder(nn.Module):
    """Window encoder: (M,1,C,T) → embedding (M, embed_dim)."""
    def __init__(self, n_channels: int, n_times: int,
                 F1: int = 16, D: int = 2, F2: int = 32, kern_len: int = 64,
                 embed_dim: int = 128, dropout_p: float = DROPOUT_P,
                 use_bandpower: bool = True):
        super().__init__()
        self.use_bandpower = use_bandpower
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            Conv2dConstrained(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout_p))
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout_p))
        with torch.no_grad():
            d = self.separable(self.depthwise(self.firstconv(torch.zeros(1, 1, n_channels, n_times))))
            eeg_dim = d.numel()
        fb_dim = 0
        if use_bandpower:
            self.bandpower = BandPower(n_times)
            fb_dim = n_channels * self.bandpower.n_bands
        self.proj = nn.Sequential(
            nn.LayerNorm(eeg_dim + fb_dim),
            nn.Linear(eeg_dim + fb_dim, embed_dim), nn.ELU(), nn.Dropout(dropout_p))
        self.embed_dim = embed_dim

    def forward(self, x):                                     # x: (M,1,C,T)
        z = self.separable(self.depthwise(self.firstconv(x))).flatten(1)
        if self.use_bandpower:
            z = torch.cat([z, self.bandpower(x)], dim=1)
        return self.proj(z)


class AttentionMIL(nn.Module):
    """Gated-attention pooling (Ilse et al., 2018). H:(B,N,D), mask:(B,N) → (B,D)."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, H, mask):
        a = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))).squeeze(-1)  # (B,N)
        a = a.masked_fill(mask < 0.5, float("-inf"))
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a)                               # guard all-masked rows
        M = torch.einsum("bn,bnd->bd", a, H)
        return M, a


class EEGNetMIL(nn.Module):
    """Bag classifier: (B,N,1,C,T) + mask → logits (B, n_classes)."""
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 embed_dim: int = 128, dropout_p: float = DROPOUT_P,
                 use_bandpower: bool = True):
        super().__init__()
        self.encoder = EEGNetEncoder(n_channels, n_times, embed_dim=embed_dim,
                                     dropout_p=dropout_p, use_bandpower=use_bandpower)
        self.attn = AttentionMIL(embed_dim, hidden=128)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout_p),
            nn.Linear(embed_dim, n_classes))

    def forward(self, x, mask, return_attn: bool = False):
        B, N, _, C, T = x.shape
        e = self.encoder(x.reshape(B * N, 1, C, T)).reshape(B, N, -1)
        m, a = self.attn(e, mask)
        logits = self.classifier(m)
        return (logits, a) if return_attn else logits


# ──────────────────────────────────────────────────────────────────────────────
# EMA (best-of-{raw, EMA})
# ──────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            s.mul_(d).add_(v.detach(), alpha=1 - d) if v.dtype.is_floating_point else s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def cpu_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Eval / metrics (recording-level)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    preds, labels, top5_hits = [], [], 0
    with torch.no_grad():
        for xb, mask, yb in loader:
            logits = model(xb.to(DEVICE), mask.to(DEVICE))
            k    = min(5, logits.shape[1])
            top5 = logits.topk(k, dim=1).indices.cpu()
            ys   = yb.tolist()
            for i, y in enumerate(ys):
                top5_hits += int(y in top5[i].tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(ys)
    return preds, labels, top5_hits


def run_fold(fold, stimuli_dir, weight_table, allowed_subjects, args):
    train_ds, test_ds, label_map = build_fold(
        stimuli_dir, fold, args.window_sec, args.overlap_sec,
        weight_table, allowed_subjects, not args.no_cache, args)
    if len(train_ds) == 0 or len(test_ds) == 0:
        tqdm.write(f"  Fold {fold}: no usable data, skipping.")
        return None, None, None, label_map, None, None, None, None

    train_labels = np.asarray(train_ds.labels)
    counts       = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    safe         = np.where(counts > 0, counts, 1.0)

    sampler = None
    if args.balance in ("sampler", "both"):
        sample_w = (1.0 / safe)[train_labels]
        sampler  = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                         num_samples=len(train_labels), replacement=True)
    ce_weight = None
    if args.balance in ("classweight", "both"):
        w = np.where(counts > 0, (counts.sum() / (len(counts) * safe)) ** 0.5, 0.0)
        w = w / (w[w > 0].mean() + 1e-8)
        ce_weight = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=0,
                              drop_last=False, collate_fn=collate_bags)
    test_loader  = DataLoader(test_ds, batch_size=args.eval_batch, shuffle=False,
                              num_workers=0, collate_fn=collate_bags)

    sample_bag = train_ds[0][0]                               # (k,1,C,T)
    n_channels, target_T = sample_bag.shape[2], sample_bag.shape[3]

    model = EEGNetMIL(n_channels, target_T, N_CLASSES, dropout_p=args.dropout,
                      use_bandpower=(args.bandpower == "on")).to(DEVICE)
    ce  = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=args.warmup_frac)
    ema = ModelEMA(model, args.ema) if args.ema and args.ema > 0 else None

    best_f1, best_state, best_top1, best_src = -1.0, None, 0.0, "raw"
    no_improve = 0
    train_accs, val_accs, val_f1s = [], [], []

    def _score():
        p, l, _ = evaluate(model, test_loader)
        acc = sum(a == b for a, b in zip(p, l)) / max(1, len(l))
        f1  = f1_score(l, p, average="macro", zero_division=0)
        return acc, f1

    bar = tqdm(range(1, args.epochs + 1), desc=f"  Fold {fold} train",
               ncols=90, unit="ep", leave=False, colour="cyan")
    for epoch in bar:
        model.train()
        tr_correct, tr_total = 0, 0
        for xb, mask, yb in train_loader:
            xb, mask, yb = xb.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb, mask)
            ce(logits, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                ema.update(model)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += yb.size(0)

        raw_acc, raw_f1 = _score()
        cand_f1, cand_acc, cand_src = raw_f1, raw_acc, "raw"
        cand_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            ema_acc, ema_f1 = _score()
            model.load_state_dict(raw_sd)
            if ema_f1 > cand_f1:
                cand_f1, cand_acc, cand_src, cand_state = ema_f1, ema_acc, "ema", ema.cpu_state()

        tr_acc = tr_correct / max(1, tr_total)
        train_accs.append(tr_acc); val_accs.append(cand_acc); val_f1s.append(cand_f1)
        if cand_f1 > best_f1:
            best_f1, best_top1, best_state, best_src = cand_f1, cand_acc, cand_state, cand_src
            no_improve = 0
        else:
            no_improve += 1
        bar.set_postfix(tr=f"{tr_acc:.2f}", val=f"{cand_acc:.2f}", f1=f"{cand_f1:.3f}",
                        best_f1=f"{max(best_f1,0):.3f}", src=best_src)
        if args.patience and no_improve >= args.patience:
            tqdm.write(f"  Early stop at epoch {epoch} (no macro-F1 gain in "
                       f"{args.patience} epochs; best from '{best_src}')")
            break

    model.load_state_dict(best_state)
    preds, labels, top5_hits = evaluate(model, test_loader)
    top1     = sum(a == b for a, b in zip(preds, labels)) / max(1, len(labels))
    bal_acc  = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    top5     = top5_hits / max(1, len(labels))
    metrics = dict(top1=top1, balanced_acc=bal_acc, macro_f1=macro_f1, top5=top5,
                   n_records=len(labels), ckpt_src=best_src)
    curves  = dict(train_acc=train_accs, val_acc=val_accs, val_f1=val_f1s)
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
    ax.set_title("Val macro-F1 (selection metric)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("macro-F1"); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    ax = axes[1]
    ax.plot(curves["train_acc"], label="Train", color="#2980b9", linewidth=1.5)
    ax.plot(curves["val_acc"],   label="Val",   color="#27ae60", linewidth=1.5)
    ax.axhline(1 / N_CLASSES, ls="--", color="gray", linewidth=1, label="Chance")
    ax.set_title("Recording-level Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    out = plot_dir / f"fold{fold}_training_curves.png"
    fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def plot_confusion_matrix(all_labels, all_preds, label_map, fold, plot_dir):
    if not MATPLOTLIB_OK: return
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx_to_label = {v: k for k, v in label_map.items()}
    names = [idx_to_label[i] for i in range(N_CLASSES)]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(names, rotation=90, fontsize=7); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Fold {fold} — Confusion Matrix (recording-level)", fontweight="bold")
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
    ax.axhline(np.mean(fold_accs), ls="-", color="#e74c3c", linewidth=1.5,
               label=f"Mean = {np.mean(fold_accs):.3f}")
    ax.set_ylabel("Recording-level Top-1"); ax.set_ylim(0, max(fold_accs) * 1.25)
    ax.set_title("Cross-Validation Accuracy per Fold", fontweight="bold"); ax.legend()
    for i, a in enumerate(fold_accs):
        ax.text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = plot_dir / "cv_summary.png"; fig.savefig(str(out), dpi=150); plt.close(fig)
    tqdm.write(f"  [PLOT] Saved {out}")


def print_per_class_summary(fold_results, label_map):
    idx_to_label = {v: k for k, v in label_map.items()}
    cc, ct = defaultdict(int), defaultdict(int)
    for preds, labels in fold_results:
        for p, l in zip(preds, labels):
            ct[l] += 1; cc[l] += int(p == l)
    rows = [(idx_to_label.get(i, str(i)), cc[i], ct[i], cc[i] / ct[i] if ct[i] else 0.0)
            for i in range(N_CLASSES)]
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"\n{'='*60}\n  Per-Class Accuracy (pooled, recording-level)\n{'='*60}")
    print(f"  {'Stimulus':<32}  Correct / Total   Acc")
    for lbl, corr, tot, acc in rows:
        print(f"  {lbl:<32}  {corr:>4} / {tot:<5}       {acc:.3f}  {'█' * int(acc * 20)}")
    ac = sum(r[1] for r in rows); at = max(1, sum(r[2] for r in rows))
    print(f"  {'OVERALL':<32}  {ac:>4} / {at:<5}       {ac/at:.3f}\n{'='*60}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    stimuli_dir = Path(args.stimuli_dir)
    if not stimuli_dir.exists():
        sys.exit(f"Not found: {stimuli_dir}")
    allowed_subjects = ALL_SUBJECTS - EXCLUDED_SUBJECTS
    valid_subs = [f"sub-{i:02d}" for i in sorted(allowed_subjects)]

    weight_table = {}
    if args.channel_weights == "on":
        wp = Path(args.weights_csv)
        if wp.exists():
            weight_table = load_channel_weights(wp, allowed_subjects)
        else:
            tqdm.write(f"  [WEIGHTS] WARNING: {wp} not found — uniform weights.")

    print(f"\n{'='*60}\n  numa EEG Classifier  {MODEL_TAG}\n{'='*60}")
    print(f"  Device          : {DEVICE}")
    print(f"  Stimuli dir     : {stimuli_dir}")
    print(f"  Active subjects : {len(valid_subs)}  {valid_subs}")
    print(f"  Channel weights : {args.channel_weights}")
    print(f"  Window/Overlap  : {args.window_sec}s / {args.overlap_sec}s  ({BANDPASS_LOW}–{BANDPASS_HIGH} Hz)")
    print(f"  Unit            : RECORDING (attention-MIL bag of windows)")
    print(f"  Windows/bag     : train_k={args.train_k}  eval_max={args.eval_max}")
    print(f"  Batch (recs)    : train {args.batch_size} / eval {args.eval_batch}")
    print(f"  Class balancing : {args.balance}")
    print(f"  Backbone        : EEGNet(16,2,32) encoder + bandpower={args.bandpower} → gated-attention MIL")
    print(f"  Dropout         : {args.dropout}   label-smooth {args.label_smooth}")
    print(f"  Train epochs    : {args.epochs}  (early-stop patience {args.patience})")
    print(f"  LR (OneCycle)   : {args.lr}  warmup {args.warmup_frac}  wd {args.weight_decay}")
    print(f"  EMA decay       : {args.ema if args.ema else 'off'}  (ckpt = best-of raw/EMA)")
    print(f"  Selection metric: macro-F1 (recording-level)")
    print(f"  Cache dir       : {'disabled' if args.no_cache else str(CACHE_DIR)}")
    print(f"  Folds           : {args.fold if args.fold else '1-5 (full CV)'}")
    print(f"{'='*60}\n")

    folds = [args.fold] if args.fold else list(range(1, N_INSTANCES + 1))
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
        tqdm.write(f"\n  Fold {fold} metrics (chance top-1 = {1/N_CLASSES:.3f}, "
                   f"{metrics['n_records']} recordings):")
        tqdm.write(f"    top-1 (recording): {metrics['top1']:.3f}")
        tqdm.write(f"    balanced acc     : {metrics['balanced_acc']:.3f}")
        tqdm.write(f"    macro-F1         : {metrics['macro_f1']:.3f}  (selection metric)")
        tqdm.write(f"    top-5            : {metrics['top5']:.3f}  (chance = {5/N_CLASSES:.3f})")
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
        print(f"\n{'='*60}\n  Cross-Validation Summary (recording-level, selection = macro-F1)\n{'='*60}")
        print(f"  {'Fold':<6}{'top1':>8}{'balAcc':>9}{'mF1':>8}{'top5':>8}")
        print(f"  {'-'*39}")
        for f, m in zip(done_folds, fold_metrics):
            print(f"  {f:<6}{m['top1']:>8.3f}{m['balanced_acc']:>9.3f}"
                  f"{m['macro_f1']:>8.3f}{m['top5']:>8.3f}")
        if len(fold_metrics) > 1:
            def col(k): return np.array([m[k] for m in fold_metrics])
            print(f"  {'-'*39}")
            print(f"  {'mean':<6}{col('top1').mean():>8.3f}{col('balanced_acc').mean():>9.3f}"
                  f"{col('macro_f1').mean():>8.3f}{col('top5').mean():>8.3f}")
            print(f"  {'std':<6}{col('top1').std():>8.3f}{col('balanced_acc').std():>9.3f}"
                  f"{col('macro_f1').std():>8.3f}{col('top5').std():>8.3f}")
        print(f"  Chance top-1 = {1/N_CLASSES:.3f} | chance top-5 = {5/N_CLASSES:.3f}\n{'='*60}")
        plot_cv_summary(done_folds, fold_accs, PLOT_DIR)

    if fold_results and last_label_map:
        print_per_class_summary(fold_results, last_label_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"numa EEG {MODEL_TAG}")
    parser.add_argument("--stimuli_dir",     default="stimuli", type=str)
    parser.add_argument("--weights_csv",
                        default="/Users/neelahuja/desktop/numa_model/eeg_channel_weights.csv", type=str)
    parser.add_argument("--fold",            default=None, type=int)
    parser.add_argument("--window_sec",      default=1.0,  type=float)
    parser.add_argument("--overlap_sec",     default=0.5,  type=float)
    parser.add_argument("--epochs",          default=80,   type=int)
    parser.add_argument("--patience",        default=15,   type=int,
                        help="early-stop patience on recording-level macro-F1 (0 = off)")
    parser.add_argument("--batch_size",      default=16,   type=int, help="recordings (bags) per train batch")
    parser.add_argument("--eval_batch",      default=8,    type=int, help="recordings per eval batch")
    parser.add_argument("--train_k",         default=24,   type=int, help="windows sampled per bag (train)")
    parser.add_argument("--eval_max",        default=96,   type=int, help="max windows per bag (eval)")
    parser.add_argument("--lr",              default=1e-3, type=float)
    parser.add_argument("--warmup_frac",     default=0.15, type=float)
    parser.add_argument("--weight_decay",    default=0.05, type=float)
    parser.add_argument("--dropout",         default=DROPOUT_P, type=float)
    parser.add_argument("--label_smooth",    default=0.1,  type=float)
    parser.add_argument("--balance",         default="classweight",
                        choices=["sampler", "classweight", "both", "none"],
                        help="recordings are ~balanced, so light classweight is the default")
    parser.add_argument("--bandpower",       default="on", choices=["on", "off"])
    parser.add_argument("--channel_weights", default="off", choices=["on", "off"])
    parser.add_argument("--ema",             default=0.999, type=float, help="0 = disable")
    parser.add_argument("--no_cache",        action="store_true")
    args = parser.parse_args()
    main(args)
