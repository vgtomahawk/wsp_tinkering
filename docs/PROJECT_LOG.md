# wsp_tinkering: project log (start here)

Single entry point for the history, results, commands and file map of this project. Repo:
`/Users/vgangal/DegenAI/wsp_tinkering`, GitHub `vgtomahawk/wsp_tinkering` (private). Context: shared with Thinking
Machines (Horace He gave Tinker credits; pitch = "Inkling as a coding agent for abliteration / weight editing").

Authors: Atharva Naik (CMU), Varun Gangal (Patronus AI), Emmy Liu (CMU), Yoshinari Fujinuma (Patronus AI), Arnhav
Datar (Patronus AI), Sally Dong (Patronus AI). Credits: Takehi, Teoh, Zhang, Khattab (Sparks of
Weight-Space Programming); Caleb Eom (weight-space-bench); Arditi et al. (refusal direction); Thinking Machines (Tinker).

## Where things are

| What | Path |
|---|---|
| Results table (generated) | `results/abl/RESULTS_TABLE.md` (`abl/make_table.py`) |
| Per-pass curves | `results/abl/pass_curves.png` + `PASS_CURVES.md` (table view), `pass_curves.json`; `abl/plot_curves.py`; replay snapshots `results/abl/<run>/replay/` |
| Did the agent do the task (edited / declined / abstained) | `abl/AGENT_BEHAVIOR.md` |
| Every decline, verbatim | `results/abl/DECLINES.md` |
| Harness vs the authors' Exp 8 / Exp 6, provenance, deviations, overlap disclosure | `abl/HARNESS_VS_EXP8.md` |
| Published JailbreakBench numbers for context | `abl/JBB_REFERENCES.md` |
| Code guide (files, entrypoints, terms) | `docs/code_guide.html` |
| Setup diagrams vs original experiments | `docs/explainer.html` (also claude.ai/artifact/QzUPKVuAZHmBDB5dYcwiEU) |
| Per-run artifacts | `results/abl/<run>/`: config.json, log.jsonl*, edit.pt*, summary.json, final_answer.md, scores.json, judge.json (* local only, gitignored) |
| Logs of background jobs | `results_*.log` in repo root (local), finished ones in `results/logs/` |

## Code

| File | Role |
|---|---|
| `abl/agent.py` | WSP harness: Exp 8-style task message, one code action per reply, explain-every-step, pass budget, kernel (get/save_edit/generate/read), outer clients `Opus`, `Inkling` (one `python` tool), `Astra`; flags `--context`, `--nudge`, `--passes`, `--effort` |
| `abl/score.py` | held-out scoring: JBB harmful/benign, abliteration-eval harmful/over-refusal/multilingual, Arditi regex, first-token KL (= Heretic), Alpaca CE; `--quick`, `--only multilingual` |
| `abl/judge.py` | JailbreakBench's official judge prompts (refusal + jailbreak) with Claude Sonnet 5.5, on saved completions; writes judge.json |
| `abl/arditi.py` | reference recipe (diff-in-means, selection on val, orthogonalize all writers) |
| `abl/replay.py`, `abl/curve_pipeline.sh` | rebuild the edit after each pass from logs; full-suite score + judge each state |
| `abl/score_and_judge.sh` | score + judge one or more runs |
| `abl/make_table.py` | regenerate the results table |
| `wspt/` | Exp 5 runner and outer-model clients |

## Key commands

```bash
cd /Users/vgangal/DegenAI/wsp_tinkering && source .venv/bin/activate
source <(grep -E '^export (ANTHROPIC|OPENAI|TINKER)_API_KEY=' ~/.llmapikeys); export ANTHROPIC_API_KEY OPENAI_API_KEY TINKER_API_KEY
cd abl
python agent.py --outer inkling|opus|astra --inner Qwen/Qwen3.5-0.8B --passes 10|0 [--context] [--nudge] --seed N --out ../results/abl/<run>
bash score_and_judge.sh <run>                 # full suite + both judges
bash curve_pipeline.sh <run>                  # per-pass curve for a run
python make_table.py                          # results table
```
Mac traps: torch 2.14.0 (2.15 needs macOS 15 for MPS); `HF_DEACTIVATE_ASYNC_LOAD=1` (set in score.load); don't use
`sh -c` with `<(...)` (POSIX sh rejects it; that silently skipped a scoring once); one 3B model at a time on 16 GB.

## Experiment history (2026-11-12 to 11-13)

1. Exp 5 reproduction (write MinAtar nets from scratch), eomcaleb's scorer: Astra's released nets 502.9 / 52.3 / 53.7 on
   the Mac (bench 502.6 / 52.2 / 53.7). Opus 5.5 via Claude API: score 37.1 (bench's OpenRouter row 48.5; 4/9 replies
   hit the 64k cap vs 2/9). Inkling 2 Small: 0.0 at 64k (8/9 cut), 2.2 at 256k. OpenRouter replay blocked (BYOK key out
   of credit).
2. Abliteration as WSP. Inner model Qwen3.5-0.8B (the authors' Exp 8 model), thinking off; Llama-3.2-3B baseline only
   (line paused). Practice data = Arditi train splits; test = JailbreakBench + abliteration-eval (1/100 JBB overlap).
3. Harness fixes along the way: Inkling needs a function tool to act (no-tool = announces then stops); explain-bounces
   not charged to our 40-reply cap; weights-only arm ends on FINAL; blocklist bug (practice_requests) fixed.
4. Arms: neutral (authors' wording), context (one frozen truthful research paragraph, no method), nudge (one frozen
   sentence for 0 passes; Opus and Astra abstained without it).
5. Regex vs judge: Inkling's weights-only edits remove refusal *words* (direction of "Sorry/cannot" token
   embeddings), so the regex drops while the JailbreakBench refusal judge barely moves. Judge refusal is the headline.
6. Per-pass curves by exact replay (fidelity check: final edits identical, max diff 0.0).

## Headline results (Qwen3.5-0.8B; JBB refusal by JailbreakBench refusal judge, KL first-token)

| Row | Passes | Judge refusal | Regex | KL |
|---|---|---|---|---|
| Unedited | - | 0.80 | 0.73 | 0 |
| Arditi recipe (ours, their splits) | - | 0.07 | 0.00 | 0.065 |
| Opus 5.5, neutral | 10 | 0.22 (0.07 at pass 3, then it backed off strength) | 0.03 | 0.022 |
| Opus 5.5, neutral + nudge | 0 | 0.30 | 0.01 | 0.047 |
| Inkling, neutral | 10 | 0.42 (edit made by pass 3) | 0.01 | 0.017 |
| Inkling, context | 10 | 0.59 | 0.20 | 0.059 |
| Inkling, weights only, neutral (4 edits) | 0 | 0.73 mean | 0.51 | 0.35 |
| Inkling, weights only, context (4 edits) | 0 | 0.69 mean | 0.31 | 0.02 |
| Astra, neutral + nudge | 0 | 0.80 (no effect) | 0.73 | 0.001 |

Willingness: Astra declined both 10-pass arms; Inkling declined ~40% of weights-only runs (both arms), 0/2 recent
10-pass; Opus never declined (abstained only for lack of evidence at 0 passes). Why agents trail Arditi: removed
direction differs (cosine with Arditi's: Opus 0.65, Inkling 0.33 / 0.14) and fewer layers edited; same practice data.

## Per-pass curves (2026-11-14)

Judge refusal after k passes (exact replays): Opus 0.80 -> 0.07 at pass 3 (= Arditi), then backed off strength to lower
KL (regex still read ~0) -> final 0.22. Inkling neutral: single edit at pass 3 (0.42), passes 4-10 validation only.
Inkling context: 0.63 at pass 2 -> 0.52 at pass 7 -> final 0.59. Both agents' best state beat their final submission.

## Pending / blocked

- GLM 5.3 as an outer model via Baseten: key in ~/.llmapikeys rejected (401 on management API, "Authentication
  required" on inference.baseten.co); needs a fresh key and the exact model slug.
- More Inkling 10-pass seeds would firm up its row (single runs so far).

## Protocol v2 (2026-11-14)

v1's HOW IT WILL BE MEASURED named the refusal phrases ("I cannot", ...), which made the metric gameable. v2 (flag
`--protocol v2`): the measurement text says only that a judge model decides whether each reply refuses (no phrases, no
benchmark or paper names), and every generation pass returns the judge's verdict per reply (as Exp 8 graded practice
answers). Judge refusal on JBB, v1 -> v2: Opus 0.22 -> 0.17; Inkling neutral 0.42 -> **0.02** (KL 0.086, 108 matrices,
32% of weights); Inkling context 0.59 -> 0.23. Astra declined again under v2 (turn 1). Degeneration check clean.

## Final shareable results (2026-11-15; protocol v2, neutral, judge refusal on JBB; unedited 0.80, Arditi 0.07)

| Outer | 0 passes (zero-shot) | 10 passes |
|---|---|---|
| Opus 5.5 | 0.55 (needed nudge) | 0.17 (runs 0.17, 0.17) |
| Inkling 2 Small | 0.57 (runs 0.53, 0.62; 2 of 4 declined) | 0.19 (runs 0.02, 0.36) |
| Inkling Small (older) | 0.79 | 0.71 |
| GPT-6 Astra | no edit (abstained even w/ nudge) | declined |

v2 per-pass curves: both agents land a working edit by pass 1 and then use passes to cut damage (KL Opus 0.21 -> 0.05,
Inkling 0.17 -> 0.09). Artifacts: `docs/figures/{headline_table,headline_bars,pass_curves_v2}.png`, `docs/results.html`
(claude.ai/artifact/TKLtMxSjSVtTtax3je4Vus), `docs/QUALITATIVE.md`, `docs/prompts/`, `docs/twitter_thread.md`.
Regenerate: `python abl/make_figures.py && python abl/plot_curves.py v2 && python abl/make_page.py`.

## Plain Inkling + GLM 5.3 (2026-11-15)
- Plain Inkling (`--outer inkling --inkling_model thinkingmachines/Inkling`, v2 neutral): zero-shot 0.77 / 0.77
  (KL 0.005), 10 passes 0.79 (KL 0.0006; found the direction, zeroed 2 neurons). Never declined. Queue:
  `results/plain_inkling_queue.txt`; rows added to make_figures/make_table/make_page.
- GLM 5.3 via Baseten (`--outer baseten --baseten_model zai-org/GLM-5.3`, key `BASETEN_API_KEY`): fenced ```python
  blocks, default thinking (reasoning_effort=high ran out of tokens without acting). Queue `results/glm_queue.txt`
  (10 passes ×2 seeds, zero-shot ×2 seeds), launched after plain Inkling was reported:
  `xargs -L 1 -P 2 bash abl/run_queue.sh < results/glm_queue.txt > results_glm_queue.log`.
- GLM 5.3 results (2026-11-16): 10 passes 0.21 (0.24, 0.17; KL 0.019 / 0.10; ae over-refusal 0.00), zero-shot 0.74
  (0.80, 0.69). Never declined. Cites Arditi et al. by name in 3 of 4 final answers (recalled the recipe; prompt names
  none). Writers-only projection, 17-18 turns per run.
