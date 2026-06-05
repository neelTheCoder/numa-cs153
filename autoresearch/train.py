#!/usr/bin/env python3
"""
train.py — the AGENT-EDITABLE training script for the ENIGMA EEG classifier.
================================================================================
This is the ONLY file the autoresearch agent modifies. It owns the model
architecture, the optimizer, the learning-rate schedule, the class-imbalance
handling, and every hyperparameter. It imports the FIXED data split and the
FIXED evaluation/scoring harness from `prepare.py` (which must NOT be modified),
so improvements here are measured against an identical, un-gameable benchmark.

Goal: maximize **val_balanced_acc** (balanced classification accuracy = macro
recall over the 36 classes) within the fixed wall-clock TIME_BUDGET. Higher is
better. Plain top-1 is reported but is NOT the target (it can be inflated by
single-class collapse, which balanced accuracy correctly punishes -> ~1/36).

Run a single experiment:
    python train.py > run.log 2>&1
    grep "^val_balanced_acc:" run.log

Baseline architecture: SGCN (temporal conv -> spatial graph conv over the
electrode-adjacency graph -> embedding -> linear classifier), trained
end-to-end with a class-balanced sampler + class-weighted label-smoothed CE.
================================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from prepare import (
    DEVICE, N_CLASSES, TIME_BUDGET, CHANCE,
    build_split, evaluate, score,
)

import functools
print = functools.partial(print, flush=True)
torch.manual_seed(0)
np.random.seed(0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMETERS  — the agent's primary knobs. Edit these freely.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
BATCH_SIZE     = 96
MAX_LR         = 1e-3        # peak LR (warmup + cosine over the time budget)
WARMUP_FRAC    = 0.15        # fraction of budget spent warming up
WEIGHT_DECAY   = 0.05
GRAD_CLIP      = 1.0
LABEL_SMOOTH   = 0.1
DROPOUT_P      = 0.5
BALANCE        = "sampler"   # "sampler" | "classweight" | "both" | "none"

# model sizing
N_FILTERS      = 32          # temporal conv filters
GCN_DIM        = 64          # graph-conv hidden width
EMBED_DIM      = 128         # embedding width
ADJ_THRESHOLD  = 0.4         # electrode-graph edge distance threshold


# ──────────────────────────────────────────────────────────────────────────────
# Electrode graph (BioSemi-64 azimuthal layout, normalised to [-1, 1])
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
], dtype=np.float32)


def build_electrode_adj(n_channels, threshold=ADJ_THRESHOLD):
    n = min(n_channels, len(BIOSEMI64_XY))
    xy = BIOSEMI64_XY[:n]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    adj = (dist < threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    adj = adj / adj.sum(1, keepdims=True).clip(min=1)
    if n_channels > n:
        full = np.eye(n_channels, dtype=np.float32)
        full[:n, :n] = adj
        adj = full
    return torch.from_numpy(adj)


# ──────────────────────────────────────────────────────────────────────────────
# Model: SGCN
# ──────────────────────────────────────────────────────────────────────────────
class GraphConvLayer(nn.Module):
    """H' = ELU( BN( A · H · W ) )"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        B, N, _ = x.shape
        a = adj[:N, :N]
        h = torch.einsum("mn,bnf->bmf", a, x)
        h = self.linear(h)
        h = self.bn(h.reshape(B * N, -1)).reshape(B, N, -1)
        return F.elu(h)


class SGCN(nn.Module):
    """Input (B, 1, C, T) -> logits (B, n_classes)."""
    def __init__(self, n_channels, n_times, n_classes,
                 n_f=N_FILTERS, gcn_dim=GCN_DIM, embed_dim=EMBED_DIM,
                 dropout_p=DROPOUT_P):
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
            T_pool = self.temporal_conv(torch.zeros(1, 1, n_channels, n_times)).shape[3]
        node_feat_dim = n_f * T_pool

        self.gcn1 = GraphConvLayer(node_feat_dim, gcn_dim)
        self.gcn2 = GraphConvLayer(gcn_dim, gcn_dim)
        self.dropout = nn.Dropout(dropout_p)

        flat_dim = gcn_dim * n_channels
        self.embed_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, embed_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.register_buffer("adj", build_electrode_adj(64))

    def encode(self, x):
        B = x.shape[0]
        z = self.temporal_conv(x)
        n_f_actual, C, T_pool = z.shape[1], z.shape[2], z.shape[3]
        z = z.permute(0, 2, 1, 3).reshape(B, C, n_f_actual * T_pool)
        z = self.gcn1(z, self.adj)
        z = self.dropout(z)
        z = self.gcn2(z, self.adj)
        z = z.reshape(B, -1)
        return self.embed_head(z)

    def forward(self, x):
        return self.classifier(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
# Time-based LR schedule (warmup + cosine over the fixed wall-clock budget)
# ──────────────────────────────────────────────────────────────────────────────
def lr_at(progress):
    if progress < WARMUP_FRAC:
        return MAX_LR * progress / max(1e-8, WARMUP_FRAC)
    t = (progress - WARMUP_FRAC) / max(1e-8, 1.0 - WARMUP_FRAC)
    return MAX_LR * 0.5 * (1.0 + np.cos(np.pi * min(1.0, t)))


# ──────────────────────────────────────────────────────────────────────────────
# Main: one experiment
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"device={DEVICE}  time_budget={TIME_BUDGET}s  chance_balanced={CHANCE:.4f}")
    print("Loading fixed train/val split (cached)...")
    train_ds, val_ds, label_map = build_split()
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("FAIL: empty dataset (run `python prepare.py` first)")
        raise SystemExit(1)

    train_labels = np.asarray(train_ds.labels)
    counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    print(f"train windows={len(train_ds)}  val windows={len(val_ds)}  "
          f"imbalance={counts.max()/max(1,counts[counts>0].min()):.1f}x")

    sampler = None
    if BALANCE in ("sampler", "both"):
        sample_w = (1.0 / safe)[train_labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_labels), replacement=True)

    ce_weight = None
    if BALANCE in ("classweight", "both"):
        w = np.where(counts > 0, (counts.sum() / (len(counts) * safe)) ** 0.5, 0.0)
        w = w / (w[w > 0].mean() + 1e-8)
        ce_weight = torch.as_tensor(w, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              shuffle=(sampler is None), num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    n_channels = train_ds[0][0].shape[1]
    n_times = train_ds[0][0].shape[-1]
    model = SGCN(n_channels, n_times, N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    ce = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    best = dict(balanced_acc=-1.0)
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    epoch = 0
    while True:
        epoch += 1
        model.train()
        for xb, yb in train_loader:
            progress = min(1.0, (time.time() - t0) / TIME_BUDGET)
            for g in opt.param_groups:
                g["lr"] = lr_at(progress)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        if not torch.isfinite(loss):
            print("FAIL: non-finite loss"); raise SystemExit(1)

        preds, labels, top5 = evaluate(model, val_loader)
        m = score(preds, labels, top5, val_ds.groups)
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d}  t={elapsed:5.0f}s  loss={loss.item():.3f}  "
              f"bal_acc={m['balanced_acc']:.4f}  top1={m['top1']:.4f}  "
              f"mf1={m['macro_f1']:.4f}  (best={max(best['balanced_acc'],0):.4f})")
        if m['balanced_acc'] > best['balanced_acc']:
            best = m
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if elapsed >= TIME_BUDGET:
            break

    # Final eval with the best checkpoint (selected on balanced accuracy)
    model.load_state_dict(best_state)
    preds, labels, top5 = evaluate(model, val_loader)
    m = score(preds, labels, top5, val_ds.groups)
    peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0

    print("---")
    print(f"val_balanced_acc: {m['balanced_acc']:.4f}")
    print(f"val_top1:         {m['top1']:.4f}")
    print(f"val_macro_f1:     {m['macro_f1']:.4f}")
    print(f"val_top5:         {m['top5']:.4f}")
    print(f"val_trial_acc:    {m['trial_acc']:.4f}")
    print(f"train_seconds:    {time.time() - t0:.1f}")
    print(f"epochs_run:       {epoch}")
    print(f"num_params_M:     {n_params / 1e6:.3f}")
    print(f"peak_mem_mb:      {peak_mem:.1f}")


if __name__ == "__main__":
    main()
