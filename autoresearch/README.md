# autoresearch — ENIGMA EEG classifier

An [autoresearch](https://github.com/miolini/autoresearch-macos)-style setup that
lets a coding agent autonomously optimize the **classification accuracy** of the
ENIGMA EEG object-decoding model.

## Files
- `prepare.py` — **fixed** infrastructure (data split + evaluation contract).
  Builds the preprocessed window cache. **Never edited by the agent.**
- `train.py` — the **agent-editable** model + optimizer + training loop. Adapted
  from `train_enigma_v3.py`. Prints a greppable metric block; the agent edits
  this file to improve `val_balanced_acc`.
- `program.md` — the agent's instructions (the loop, the rules, the metric).
- `results.tsv` — experiment ledger (one row per experiment).
- `run_autoresearch.sh` — launches the agent (Claude Code, Sonnet).

## Metric
Primary target = **`val_balanced_acc`** (balanced/macro accuracy over 36 classes,
higher is better; chance ≈ 0.0278). Honest version of "classification accuracy"
on this imbalanced task. Also reported: top-1, macro-F1, top-5, trial-level.

## Quick start
```bash
cd autoresearch
../venv/bin/python prepare.py        # one-time: build the window cache
../venv/bin/python train.py          # baseline: ~8 min, prints metric block
./run_autoresearch.sh                # launch the autonomous Sonnet agent
```
Each experiment trains for a fixed `TIME_BUDGET` (8 min) and is hard-capped at
10 min. Stop the agent with Ctrl-C; the best result lives on the
`autoresearch/<tag>` git branch and in `results.tsv`.
