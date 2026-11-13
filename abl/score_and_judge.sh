#!/bin/bash
# Usage: bash abl/score_and_judge.sh <tag> [<tag> ...]   (needs ANTHROPIC_API_KEY exported for the judge step)
set -eu
cd "$(dirname "$0")"
for t in "$@"; do
  ../.venv/bin/python score.py --model Qwen/Qwen3.5-0.8B --tag "$t" --base_tag qwen35_08b_base --edit "../results/abl/$t/edit.pt"
  ../.venv/bin/python judge.py --tags "$t"
done
echo ALL_DONE
