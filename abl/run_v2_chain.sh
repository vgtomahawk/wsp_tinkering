#!/bin/bash
# Usage: bash abl/run_v2_chain.sh "<outer> <arm-flag-or-none> <run-name>" ...   (keys exported by the caller)
set -u
cd "$(dirname "$0")"
for spec in "$@"; do
  set -- $spec; outer=$1; arm=$2; run=$3
  flag=""; [ "$arm" = "context" ] && flag="--context"
  ../.venv/bin/python agent.py --outer $outer --inner Qwen/Qwen3.5-0.8B --passes 10 --turns 40 --seed 0 --protocol v2 $flag \
    --out ../results/abl/$run > ../results_$run.log 2>&1
  n=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tensors']))" ../results/abl/$run/summary.json 2>/dev/null || echo 0)
  if [ "$n" -gt 0 ]; then bash score_and_judge.sh $run > ../results_sj_$run.log 2>&1; fi
  echo "CHAIN_STEP_DONE $run edited_tensors=$n"
done
