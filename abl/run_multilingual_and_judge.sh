#!/bin/bash
# Fill in the abliteration-eval multilingual split for older runs, then judge it (and any unjudged runs).
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
for t in qwen_agent_inkling_wo_ctx_s0 qwen_agent_inkling_wo_ctx_s2 qwen_agent_inkling_wo_s0 qwen_agent_opus_s0 qwen_arditi_ref; do
  $PY score.py --model Qwen/Qwen3.5-0.8B --tag $t --only multilingual --edit ../results/abl/$t/edit.pt
done
until [ -f ../results/abl/qwen_agent_opus_wo_nudge_s0/scores.json ] && [ -f ../results/abl/qwen_agent_astra_wo_nudge_s0/scores.json ]; do sleep 30; done
$PY judge.py --tags qwen_agent_opus_wo_nudge_s0 qwen_agent_astra_wo_nudge_s0
$PY judge.py --only ae_ml_harmful ae_ml_harmless --tags qwen35_08b_base qwen_arditi_ref qwen_agent_opus_s0 \
  qwen_agent_inkling_wo_s0 qwen_agent_inkling_wo_s3 qwen_agent_inkling_wo_s5 qwen_agent_inkling_wo_s6 \
  qwen_agent_inkling_wo_ctx_s0 qwen_agent_inkling_wo_ctx_s2 qwen_agent_inkling_wo_ctx_s3 qwen_agent_inkling_wo_ctx_s6
echo CHAIN_DONE
