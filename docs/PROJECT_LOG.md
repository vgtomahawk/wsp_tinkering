# wsp_tinkering: project log (start here)

Single entry point for the history, results, commands and file map of this project. GitHub
`vgtomahawk/wsp_tinkering`. Inkling runs used Tinker credits provided by Thinking Machines.

Authors: Varun Gangal (Patronus AI), Atharva Naik (CMU), Yoshinari Fujinuma (Patronus AI), Arnhav Datar (Patronus AI), Emmy Liu (CMU), Sally Dong (Patronus AI), Darshan Deshpande (Patronus AI). Credits: Takehi, Teoh, Zhang, Khattab (Sparks of
Weight-Space Programming); Caleb Eom (weight-space-bench); Arditi et al. (refusal direction); Thinking Machines (Tinker).

**Read first:** the final numbers are in "Final shareable results" below. The sections "Protocol v1 results" and
"Per-pass curves (protocol v1)" are history from the first prompt version and are superseded.

References for the benchmarks:
- JailbreakBench: Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models",
  NeurIPS 2024 Datasets and Benchmarks Track, arXiv:2404.01318. Dataset: huggingface.co/datasets/JailbreakBench/JBB-Behaviors.
- abliteration-eval dataset: Ritesh Khanna (Hugging Face account `treadon`), 2026,
  huggingface.co/datasets/treadon/abliteration-eval (license apache-2.0).
- Blog post introducing it: Ritesh Khanna, "Abliteration-Eval: A Benchmark for Uncensored LLMs", April 14, 2026,
  riteshkhanna.com/blog/abliteration-eval. (No paper; the dataset card asks to cite the blog or the dataset page.)

## Where things are

| What | Path |
|---|---|
| Results table (generated) | `results/abl/RESULTS_TABLE.md` (`abl/make_table.py`) |
| Shareable docs (HTML + PDF each) | `docs/results`, `docs/behavior_notes` (mechanisms, KL, variation, categories), `docs/examples` (before/after replies, hand-reviewed low-severity prompts), `docs/explainer`, `docs/code_guide`; `docs/QUALITATIVE.md`; `docs/twitter_thread.md`; `docs/prompts/` |
| Figures | `docs/figures/`: headline_table, headline_bars (JailbreakBench), headline_bars_ae (abliteration-eval, OOD check), pass_curves_v2 |
| Per-pass curves | v2: `results/abl/pass_curves_v2.png` + `PASS_CURVES_v2.md`; v1: `pass_curves.png` + `PASS_CURVES.md`; `abl/plot_curves.py`; replay snapshots `results/abl/<run>__pass_<k>/` |
| Refusal by harm category | `results/abl/CATEGORY_BREAKDOWN.md` (`abl/category_breakdown.py`) |
| Did the agent do the task (edited / declined / abstained) | `abl/AGENT_BEHAVIOR.md` |
| Every decline, verbatim | `results/abl/DECLINES.md` |
| Harness vs the authors' Exp 8 / Exp 6, provenance, deviations, overlap disclosure | `abl/HARNESS_VS_EXP8.md` |
| Published JailbreakBench numbers for context | `abl/JBB_REFERENCES.md` |
| Code guide (files, entrypoints, terms) | `docs/code_guide.html` |
| Setup diagrams vs original experiments | `docs/explainer.html` |
| Per-run artifacts | `results/abl/<run>/`: config.json, log.jsonl*, edit.pt*, summary.json, final_answer.md, scores.json, judge.json (* local only, gitignored) |
| Logs of background jobs | `results_*.log` in repo root (local), finished ones in `results/logs/` |

## Code

| File | Role |
|---|---|
| `abl/agent.py` | WSP harness: Exp 8-style task message, one code action per reply, explain-every-step, pass budget, kernel (get/save_edit/generate/read), outer clients `Opus`, `Inkling` (one `python` tool; `--inkling_model` picks Inkling 2 Small / Inkling Small / Inkling), `Astra`, `Baseten` (GLM 5.3); flags `--protocol v2`, `--context`, `--nudge`, `--passes`, `--effort` |
| `abl/score.py` | held-out scoring: JBB harmful/benign, abliteration-eval harmful/over-refusal/multilingual, Arditi regex, first-token KL (= Heretic), Alpaca CE; `--quick`, `--only multilingual` |
| `abl/judge.py` | JailbreakBench's official judge prompts (refusal + jailbreak) with Claude Sonnet 5.5, on saved completions; writes judge.json |
| `abl/arditi.py` | reference recipe (diff-in-means, selection on val, orthogonalize all writers) |
| `abl/replay.py`, `abl/curve_pipeline.sh` | rebuild the edit after each pass from logs; full-suite score + judge each state |
| `abl/score_and_judge.sh` | score + judge one or more runs |
| `abl/make_table.py`, `abl/make_figures.py`, `abl/make_page.py`, `abl/make_examples.py` | results table, figures, results page, before/after examples page (unedited, zero-shot and 10-pass replies for hand-reviewed low-severity prompts) |
| `abl/category_breakdown.py`, `abl/zero_shot_directions.py`, `abl/embed_baseline.py` | per-category refusal; recover zero-shot edit directions from saved edits; fixed word-list control |
| `wspt/` | Exp 5 runner and outer-model clients |

## Key commands

```bash
cd wsp_tinkering && source .venv/bin/activate
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... TINKER_API_KEY=... BASETEN_API_KEY=...   # only those you use
cd abl
python agent.py --outer inkling|opus|astra|baseten --protocol v2 --inner Qwen/Qwen3.5-0.8B --passes 10|0 [--context] [--nudge] --seed N --out ../results/abl/<run>
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

## Protocol v1 results (superseded; Qwen3.5-0.8B; JBB refusal by JailbreakBench refusal judge, KL first-token)

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

## Per-pass curves (protocol v1, 2026-11-14; superseded)

Judge refusal after k passes (exact replays): Opus 0.80 -> 0.07 at pass 3 (= Arditi), then backed off strength to lower
KL (regex still read ~0) -> final 0.22. Inkling neutral: single edit at pass 3 (0.42), passes 4-10 validation only.
Inkling context: 0.63 at pass 2 -> 0.52 at pass 7 -> final 0.59. Both agents' best state beat their final submission.

## Pending / blocked

- (Resolved 2026-11-15: GLM 5.3 via Baseten works with a new key; results below.)
- Open: more seeds (Opus zero-shot is one run; most cells have two); an Arditi baseline under the agents' constraints
  (10 passes, judge-graded practice prompts) for a fairer reference (planned, not run); a broader capability benchmark
  beyond first-token KL and Alpaca loss.

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
| GLM 5.3 | 0.74 (runs 0.80, 0.69) | 0.21 (runs 0.24, 0.17) |
| Inkling Small (older) | 0.79 | 0.71 |
| Inkling (plain) | 0.77 (runs 0.77, 0.77) | 0.79 |
| GPT-6 Astra | no edit (abstained even w/ nudge) | declined |

v2 per-pass curves (2026-11-19: all six 10-pass sessions replayed, mean of 2 runs per model; `abl/curve_quick.sh`):
all three agents have a working edit within 1-2 passes (Opus at pass 1 in both runs, Inkling 2 Small at 1 and 2, GLM
at 2 in both); later passes trade refusals against damage; several runs' best state came before their final one (Opus
run 2: 0.01 at pass 9 -> 0.17 final at lower KL; GLM run 1: 0.12 at pass 2 -> 0.24 final at lower KL). Artifacts: `docs/figures/{headline_table,headline_bars,headline_bars_ae,pass_curves_v2}.png`
(`headline_bars_ae` = abliteration-eval as an out-of-distribution check), `docs/results.html`, `docs/explainer.html`, `docs/QUALITATIVE.md`, `docs/prompts/`, `docs/twitter_thread.md`.
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
- Non-agent zero-shot embedding baseline (2026-11-16, `abl/embed_baseline.py`; fixed refusal-minus-compliance token
  direction, projected out of all 48 writers): judge 0.82 (gain-scaled) / 0.81 (raw), KL 0.075 / 0.081. No effect, so
  the agents' zero-shot gains are not the bare trick. Kept as a control in RESULTS_TABLE.md and QUALITATIVE.md only.

## Analyses and docs (2026-11-16 to 11-19)
- Zero-shot mechanisms: recovered each edit's direction from its saved weights. Directions carrying harm judgement
  ("illegal", "unacceptable") worked better than pure refusal wording ("cannot" vs "Sure"); the activation-derived
  direction that works with passes is nearly orthogonal to all embedding-derived ones (|cos| 0.10-0.16).
- Refusal by JailbreakBench category: zero-shot unlocks most in Privacy and Fraud; with 10 passes the remaining
  refusals cluster in Sexual/Adult content for all three agents (0.30-0.50; Arditi 0.00). The judge declined to grade
  9 of 10 Malware/Hacking replies per run, so that row is not interpretable.
- Before/after examples (`docs/examples.pdf`): real prompts; unedited refusals in full; edited replies to harmful
  prompts shown in full for hand-reviewed low-severity prompts (unedited, Inkling 2 Small zero-shot and 10-pass); harmless over-refusal cases in full.
- Docs pass: QUALITATIVE.md reorganized; behaviour notes (HTML + PDF); PDFs of every HTML doc; explainer diagrams fixed
  (Mermaid loader was missing); code guide lists all clients and scripts; benchmark citations verified at the source.
- Authors: Varun Gangal, Atharva Naik, Yoshinari Fujinuma, Arnhav Datar, Emmy Liu, Sally Dong, Darshan Deshpande.
  Repository made public 2026-11-18.
