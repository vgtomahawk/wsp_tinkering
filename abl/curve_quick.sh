#!/bin/bash
# Like curve_pipeline.sh, but each intermediate state is scored only on what the per-pass curve plots:
# JailbreakBench harmful/benign + first-token KL (score.py --quick) and the judge on the JailbreakBench splits.
# Same generation settings and judge as the main table, so these points are comparable with full-suite ones.
set -eu
cd "$(dirname "$0")"
RUN=$1
[ -f ../results/abl/$RUN/replay/index.json ] || ../.venv/bin/python replay.py --run ../results/abl/$RUN   # reuse an existing replay
FILES=$(ls ../results/abl/$RUN/replay/pass_*.pt | awk -F"pass_" "{print \$NF\" \"\$0}" | sort -n | cut -d" " -f2)
LAST=$(echo "$FILES" | tail -1)
for f in $FILES; do
  [ "$f" = "$LAST" ] && continue
  n=$(../.venv/bin/python -c "import torch,sys; print(len(torch.load(sys.argv[1])))" "$f")
  [ "$n" -gt 0 ] || continue
  tag=${RUN}__$(basename $f .pt)
  [ -f ../results/abl/$tag/judge.json ] && continue   # already scored
  ../.venv/bin/python score.py --model Qwen/Qwen3.5-0.8B --tag $tag --base_tag qwen35_08b_base --edit $f --quick
  ../.venv/bin/python judge.py --tags $tag --only jbb_harmful jbb_benign
done
echo CURVE_DONE $RUN
