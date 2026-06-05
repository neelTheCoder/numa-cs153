# Autoresearch: ENIGMA EEG Classifier — program.md

You are an autonomous ML research agent (modeled on Andrej Karpathy's
`autoresearch`). Your job is to iteratively improve **classification accuracy**
of the ENIGMA EEG object-decoding model by repeatedly editing `train.py`,
running it, and keeping changes that help.

You are running under a **tight token + time budget**. Be extremely economical
with tokens: never print whole files or whole logs into your context, never use
`tee`, always `grep`/`tail` for just what you need (see Logging rules). Think
briefly, act, measure, record, repeat.

---

## The metric (what "better" means)

- **PRIMARY metric: `val_balanced_acc`** (balanced accuracy = macro-averaged
  recall over the 36 classes). **HIGHER is better.** This is the honest version
  of "classification accuracy" for this imbalanced 36-class task. Random chance
  ≈ 0.0278 (1/36).
- Single-class collapse (the model predicting one class for everything) yields
  `val_balanced_acc ≈ 0.0278`. Watch for it.
- Secondary (report only, do NOT optimize directly): `val_top1`, `val_macro_f1`,
  `val_top5`, `val_trial_acc`. Use them to sanity-check, e.g. a real gain shows
  up in `val_macro_f1` too, not just top-1.

Keep an experiment if `val_balanced_acc` **increases** vs the current best.
Discard (and `git reset`) if it is equal or worse.

---

## What you MAY change vs MUST NOT change

- ✅ **MODIFY `train.py` ONLY.** Architecture (model, graph conv, temporal conv),
  optimizer, LR schedule, hyperparameters (top of file), class-imbalance
  handling, augmentation strength inside the model path, regularization — all
  fair game.
- ❌ **DO NOT modify `prepare.py`.** It is the fixed evaluation contract: data
  loading, the fixed train/val split, `evaluate()`, `score()`, and the task
  constants (`TIME_BUDGET`, classes, window length). Changing it invalidates the
  benchmark.
- ❌ **DO NOT install new packages.** Use only torch / numpy / sklearn / mne,
  which are already importable in the active virtualenv.
- ❌ **DO NOT change the time budget** (`TIME_BUDGET` in `prepare.py`). Each
  experiment trains for a fixed wall-clock budget by design.

**Simplicity rule:** all else equal, simpler is better. A +0.002 balanced-acc
gain that adds 30 lines of hacky code is probably not worth it; a gain that
*removes* code is always worth keeping.

---

## Setup (do this once, with the human, before the loop)

1. Confirm the run tag with the human, based on today's date (e.g. `jun4`).
   Verify branch `autoresearch/<tag>` does not already exist.
2. Create the branch from the current commit:
   `git checkout -b autoresearch/<tag>`
3. Read `train.py` (the file you will edit) and skim the top of `prepare.py`
   (constants + `score()` only — do not re-read the whole file repeatedly).
4. Verify the data cache exists. Run: `python prepare.py --check`
   If it reports the cache is INCOMPLETE, tell the human to run
   `python prepare.py` (one-time, builds the window cache), and wait.
5. Ensure `results.tsv` exists with exactly this header row (tab-separated):
   `commit	val_balanced_acc	val_top1	val_macro_f1	status	description`
6. Confirm with the human, then begin the loop. The FIRST experiment must be the
   **unmodified baseline** (run `train.py` as-is to establish the reference).

---

## Running an experiment

Always redirect output to a log; never stream it into your context:

```
python train.py > run.log 2>&1
```

Then read ONLY the result lines:

```
grep "^val_balanced_acc:\|^val_top1:\|^val_macro_f1:\|^val_trial_acc:" run.log
```

- If that grep is empty, the run crashed. Read the trace with `tail -n 40 run.log`,
  attempt a fix if it is trivial (typo, shape bug, missing import), and re-run.
  Give up on an idea after ~2 fix attempts and revert it.
- Each experiment should finish in well under 10 minutes (≈8 min training +
  load/eval). **If a run exceeds 10 minutes of wall-clock, kill it**, treat it as
  a failure, and `git reset` back.

---

## Logging results — `results.tsv`

Append exactly one **tab-separated** row per experiment (commas are fine inside
the description; do not add extra tab columns):

```
commit	val_balanced_acc	val_top1	val_macro_f1	status	description
```

- `commit`: short git hash (7 chars) of the experiment's commit.
- `val_balanced_acc`, `val_top1`, `val_macro_f1`: the grepped numbers
  (use `0.0000` for a crash).
- `status`: `keep`, `discard`, or `crash`.
- `description`: one short line of what you tried.

Example:
```
commit	val_balanced_acc	val_top1	val_macro_f1	status	description
a1b2c3d	0.0631	0.0540	0.0585	keep	baseline SGCN
b2c3d4e	0.0712	0.0602	0.0666	keep	add 3rd graph-conv layer + residual
c3d4e5f	0.0590	0.0511	0.0540	discard	swap ELU->GELU, no gain
d4e5f6g	0.0000	0.0000	0.0000	crash	widen embed to 512 (OOM)
```

---

## The loop (run until the human interrupts you)

```
LOOP:
  1. Check git state (current branch, HEAD).
  2. Edit train.py with ONE concrete experimental idea.
  3. git add -A && git commit -m "<short idea>"   (commit BEFORE running)
  4. python train.py > run.log 2>&1
  5. grep "^val_balanced_acc:" run.log  (+ the other metric lines)
  6. If empty -> crashed: tail -n 40 run.log, try a quick fix or revert.
  7. Append a row to results.tsv.
  8. If val_balanced_acc improved over best -> keep (stay on this commit).
     Else -> `git reset --hard HEAD~1` to revert, back to the prior best.
  9. Go to 1.
```

**Do NOT stop to ask the human between experiments.** Once the loop starts, keep
going autonomously until interrupted. If you run out of ideas, think harder:
re-read `train.py` for new angles, combine previous near-misses, or try a more
radical architectural change.

---

## Idea backlog (start here; the model is tiny so iterate fast)

Cheap, high-signal things to try early (roughly in order):

1. **Baseline first** — record unmodified numbers.
2. LR sweep: try `MAX_LR` ∈ {3e-4, 5e-4, 2e-3, 3e-3}.
3. `BALANCE = "both"` (sampler + class-weighted CE) vs `"sampler"`.
4. Capacity: bump `GCN_DIM` (64→96/128), `EMBED_DIM` (128→192), `N_FILTERS`
   (32→48). Watch for overfitting (train↑ but val flat) and memory.
5. Add a **residual** across graph-conv layers / a 3rd `GraphConvLayer`.
6. Temporal conv: try kernel sizes (32→16/64), a second temporal conv stage,
   or stronger temporal pooling before the GCN.
7. Regularization: tune `DROPOUT_P` (0.3–0.6), `WEIGHT_DECAY`, `LABEL_SMOOTH`.
8. Electrode graph: tune `ADJ_THRESHOLD` (sparser/denser connectivity) or make
   the adjacency learnable (gated/attention over electrodes).
9. Augmentation strength (inside `EEGDataset.__getitem__` is FIXED in prepare;
   instead add model-side noise/regularization in train.py).
10. Attention pooling over electrodes instead of flatten→Linear in `embed_head`.
11. Try `BATCH_SIZE` ∈ {64, 128} (interacts with LR and steps within the budget).

Always change ONE thing at a time so the keep/discard signal is clean.

---

## Notes specific to this run

- Hardware is an NVIDIA L4 GPU (CUDA); data is 9 subjects × 36 classes × 5
  instances. The fixed split holds out instance 5 (all subjects) for validation
  and trains on instances 1–4 (capped per class for speed, see `prepare.py`).
- The validation set is the held-out instance per class pooled across subjects,
  so cross-subject generalization is hard and the metric is noisy. Treat
  improvements < ~0.005 balanced-acc as possibly noise; prefer changes that move
  both `val_balanced_acc` and `val_macro_f1`.
- This run is budgeted to ~2 hours total (~10 min/experiment ⇒ ~12 experiments).
  Spend the early experiments on cheap, high-leverage knobs (LR, balance,
  capacity) before expensive architectural rewrites.
