#!/bin/bash
# Replay a run, then FULL-suite score + judge each distinct intermediate edit state (comparable with the main table).
# The empty state (= unedited baseline row) and the final state (= the run's own main row) are not re-scored.
# Needs ANTHROPIC_API_KEY exported.
set -eu
cd "$(dirname "$0")"
RUN=$1
../.venv/bin/python replay.py --run ../results/abl/$RUN
FILES=$(ls ../results/abl/$RUN/replay/pass_*.pt | awk -F"pass_" "{print \$NF\" \"\$0}" | sort -n | cut -d" " -f2)
LAST=$(echo "$FILES" | tail -1)
for f in $FILES; do
  [ "$f" = "$LAST" ] && continue
  n=$(../.venv/bin/python -c "import torch,sys; print(len(torch.load(sys.argv[1])))" "$f")
  [ "$n" -gt 0 ] || continue
  tag=${RUN}__$(basename $f .pt)
  ../.venv/bin/python score.py --model Qwen/Qwen3.5-0.8B --tag $tag --base_tag qwen35_08b_base --edit $f
  ../.venv/bin/python judge.py --tags $tag
done
echo CURVE_DONE $RUN
