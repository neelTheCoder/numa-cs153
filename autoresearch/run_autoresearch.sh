#!/usr/bin/env bash
# Launch the autoresearch agent (Claude Code, Sonnet) on the ENIGMA EEG task.
# Run this ON the remote GPU box (enigma-train), inside a tmux session.
#
#   tmux new -s autoresearch
#   cd ~/autoresearch && ./run_autoresearch.sh
#
# Token economy: uses Sonnet (cheaper/faster than Opus) for the iteration loop,
# as requested. The loop itself is defined by program.md.
set -uo pipefail
cd "$(dirname "$0")"

# Make `claude` (installed via nvm) and system python available.
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

echo "python -> $(command -v python3)"
echo "claude -> $(command -v claude || echo NOT-FOUND)"
python3 prepare.py --check

# Auth must already be configured (see steps below). bypassPermissions lets the
# loop edit/run/commit without stopping to confirm between every experiment.
exec claude \
  --model sonnet \
  --permission-mode bypassPermissions \
  "Read program.md and kick off a new autoresearch experiment. Do the setup first (pick a run tag from today's date), then run the experiment loop autonomously: edit train.py, commit, run 'python3 train.py > run.log 2>&1', grep the metric, log to results.tsv, keep-or-revert, repeat. Use python3 (not uv). Stop after about 2 hours of wall-clock or when I interrupt you. Keep going without asking me between experiments."
