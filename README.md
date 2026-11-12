# wsp_tinkering

Reproduce experiments from *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 2026;
[blog](https://rikiyat.github.io/blog/weight-space-programming/)) with outer models we can run (Claude Opus 5.5,
later Inkling 2 Small), anchored to published numbers, then extend to new targets (e.g. abliteration).

## Foundations (git submodules, pinned)

| Path | What | License |
|---|---|---|
| `third_party/lm_understand-logs` | the authors' full logs, prompts, released networks and edits, Exp 8 eval code | CC BY 4.0 (logs/data) |
| `third_party/weight-space-bench` | eomcaleb's benchmark: Exp 5 (write MinAtar networks from scratch) with a scorer validated against Astra's released networks, a published Opus 5.5 row, and an Exp 8 math-edit harness (Linux GPU) | MIT; MinAtar from PyPI is GPL-3.0 |

Nothing in `third_party/` is edited. Our code is in `wspt/`.

## Anchors we match against

| Experiment | Reference | Source |
|---|---|---|
| Exp 5 scorer | Astra's released networks: 517.4 / 53.1 / 53.2 (paper) | lm_understand-logs |
| Exp 5, Opus 5.5 | score 48.5; Breakout 17.5, Seaquest 2.9, Asterix 27.0 (3 attempts x 1,000 episodes, effort high, 64k output, via OpenRouter) | weight-space-bench leaderboard, 2026-11-03 |
| Exp 8 (Qwen3.5-0.8B math edit) | MATH-500 7.9 -> 48.1 (50 passes, Astra); no published Opus number | lm_understand-logs `exp8_math/eval/*/merged.json` |

## Setup and runs (Mac, CPU)

```bash
git clone --recurse-submodules <this repo> && cd wsp_tinkering
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r third_party/weight-space-bench/requirements.txt anthropic
source <(grep '^export ANTHROPIC_API_KEY=' ~/.llmapikeys)

# 1. scorer check: Astra's networks (~90 s on an M1 Pro)
cd third_party/weight-space-bench && python -m weight_space_bench reference && cd -
# 2. Opus 5.5 via the Claude API, same protocol as the bench's Opus row
python -m wspt.run_minatar run claude-opus-5.5-direct
```

`config/models.toml` lists our outer models. `api = "anthropic"` sends a model to the Claude API directly
(`wspt/anthropic_direct.py`: adaptive thinking, `output_config.effort`, streaming, no tools, no refusal fallback);
other entries go through the bench's OpenRouter client unchanged.

## Log

| Date | Run | Result |
|---|---|---|
| 2026-11-12 | scorer reference on M1 Pro | Astra 502.9 / 52.3 / 53.7 (bench: 502.6 / 52.2 / 53.7; paper 517.4 / 53.1 / 53.2) |
| 2026-11-12 | Exp 5, Opus 5.5 via Claude API (effort high, 64k, 3x3) | **score 37.1** (bench's OpenRouter row: 48.5). Breakout 12.9 / **502.0** / 0.5 (mean 171.8, 690% PPO); Seaquest 0.1 x3 (all 3 hit the 64k cap, no weights); Asterix 12.4 / 21.2 / 28.7 (mean 20.8, 37% PPO). 4/9 replies hit 64k (bench: 2/9). Cost $9.42 |
