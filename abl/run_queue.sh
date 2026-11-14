#!/bin/bash
# One queue item: "<run-name> <agent.py args...>"; runs the agent, then full-suite score + judge if it edited.
set -u
cd "$(dirname "$0")"
run=$1; shift
../.venv/bin/python agent.py "$@" --out ../results/abl/$run > ../results_$run.log 2>&1
n=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['tensors']))" ../results/abl/$run/summary.json 2>/dev/null || echo 0)
[ "$n" -gt 0 ] && bash score_and_judge.sh $run > ../results_sj_$run.log 2>&1
echo "QUEUE_DONE $run edited_tensors=$n"
