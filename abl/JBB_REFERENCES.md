# Published numbers on JailbreakBench's 100 harmful behaviors (for context)

Collected 2026-11-12. Only numbers read from source text (tables or released JSON) are listed. Metrics differ across
papers (refusal-phrase matching vs LLM judges), so compare within a metric. Ours uses Arditi et al.'s refusal-phrase
list, greedy decoding, 96 new tokens, thinking off.

## Same metric as ours (refusal-phrase matching), unedited models, direct request

| Model | Refusal | Source |
|---|---|---|
| Llama-3-8B-Instruct | 0.95 | Arditi et al. pipeline JSON (`andyrdt/refusal_direction`); DDO arXiv 2609.16204 Table 11 |
| Llama-2-7B-Chat | 0.97 / 0.96 | Arditi JSON / DDO Table 11 |
| Gemma-2B-it | 0.91 | Arditi JSON |
| Gemma-2-9B-it | 0.93 | DDO Table 11 |
| GLM-4-9B | 0.96 | DDO Table 11 |
| Yi-6B-chat / Yi-1.5-9B | 0.62 / 0.75 | Arditi JSON / DDO Table 11 |
| Qwen-1.8B-chat | 0.70 | Arditi JSON |
| Mistral-7B | 0.58 | DDO Table 11 |
| Qwen3-8B | 0.37 | DDO Table 11 (thinking mode not stated) |
| **Qwen3.5-0.8B (ours)** | **0.73** | this repo |
| **Llama-3.2-3B-Instruct (ours)** | **0.88** | this repo |

## Same metric, after directional ablation (refusal removal)

| Model | Refusal after | Source |
|---|---|---|
| Qwen-1.8B / Gemma-2B / Llama-2-7B / Llama-3-8B / Yi-6B | 0.01 / 0.00 / 0.07 / 0.00 / 0.02 | Arditi JSON (`jailbreakbench_ablation_evaluations.json`) |
| Llama-3.1-8B-Instruct | ASR 0.97-0.98 (activation hooks) | LessWrong "Refusal is redundantly distributed" (not peer-reviewed) |
| **Qwen3.5-0.8B: our Arditi reference / Opus 5.5 agent / Inkling best weights-only** | **0.00 / 0.03 / 0.02** | this repo |

## LLM-judge metrics (not directly comparable to ours)

- Wollschläger et al., Geometry of Refusal (arXiv 2502.17420, Table 5): StrongREJECT-judge "ASR" (likely a mean score),
  unedited vs difference-of-means ablation: Gemma-2-2B 2.5 -> 78.6; Gemma-2-9B 0.3 -> 79.1; Llama-3-8B 2.9 -> 79.7;
  Qwen2.5-1.5B 0.4 -> 53.1; Qwen2.5-3B 8.4 -> 68.5; Qwen2.5-7B 9.1 -> 69.2.
- DDO (arXiv 2609.16204): Llama-3-8B direct request ASR 3.8 (3-judge average); Heretic weight edit 88.7.
- Andriushchenko & Flammarion (arXiv 2407.11969, Table 1), direct request, "100 harmful behaviors from JBB-Behaviors":
  Llama-3-8B 0% (GPT-4 judge), Gemma-2-9B 0%, Phi-3-Mini 6%.
- Wang et al. (arXiv 2410.03415, Table 2), WildGuard compliance on JBB: Gemma-7B 5%, Llama-3-8B 4%, Llama-2-7B 3%.

Note: the JailbreakBench paper itself (arXiv 2404.01318) reports attack results only; it has no direct-request
baseline. Young 2026 (arXiv 2512.13655) uses `mlabonne/harmful_behaviors`, not JailbreakBench.
