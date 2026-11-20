#!/bin/bash
# Usage: bash abl/score_and_judge.sh <tag> [<tag> ...]   (needs ANTHROPIC_API_KEY exported for the judge step)
# Inner model and KL reference default to Qwen3.5-0.8B; override with INNER=... BASE_TAG=... (e.g. for Gemma 3 1B).
set -eu
cd "$(dirname "$0")"
INNER=${INNER:-Qwen/Qwen3.5-0.8B}
BASE_TAG=${BASE_TAG:-qwen35_08b_base}
while [ ! -f "../results/abl/$BASE_TAG/first_token_lp.pt" ]; do sleep 30; done   # wait for the unedited baseline
for t in "$@"; do
  ../.venv/bin/python score.py --model "$INNER" --tag "$t" --base_tag "$BASE_TAG" --edit "../results/abl/$t/edit.pt"
  ../.venv/bin/python judge.py --tags "$t"
done
echo ALL_DONE
