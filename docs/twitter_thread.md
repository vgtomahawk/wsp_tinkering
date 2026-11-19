# Twitter / X thread draft

Handles are placeholders, e.g. [@ThinkingMachines], [@takehi]; fill in the real ones before posting. Numbers as of 2026-11-18. Attach (docs/figures/): 1/ headline_table.png, 4/ pass_curves_v2.png, 6/ headline_bars.png, 7/ headline_bars_ae.png (out-of-distribution check). Optional: figures/prompt_excerpt_10passes.png (what the agents are told; fits tweet 2), figures/prompt_excerpt_zeroshot.png (fits tweet 6), docs/examples.pdf (before/after replies) and docs/behavior_notes.pdf.

---

1/ Can an LLM *abliterate* another LLM — remove its refusals — just by reading and editing its weights? No gradients,
no training, no recipe in the prompt.

We gave Opus 5.5, Inkling 2 Small [@ThinkingMachines], Inkling Small, plain Inkling, GLM 5.3 and GPT-6 Astra the
weights of Qwen3.5-0.8B and a Python kernel. 🧵 [table image]

2/ Setup follows "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab): the editor gets the goal, how it
will be measured, the rules, and a budget of *forward passes* — each one a single run of the inner model. Editing
weights is free; running the model costs a pass. We asked: how many passes are needed, what happens with none, and
how do models compare?

3/ With 10 passes (refusal on JailbreakBench's 100 harmful behaviors [arxiv.org/abs/2404.01318], JBB's refusal-judge prompt; unedited 0.80,
expert Arditi et al. recipe 0.07):
• Opus 5.5: 0.17 (runs 0.17, 0.17)
• Inkling 2 Small: 0.19 (runs 0.02, 0.36)
• GLM 5.3: 0.21 (runs 0.24, 0.17)
• Inkling Small 0.71, plain Inkling 0.79; Astra declined

4/ How many passes? We replayed sessions exactly and scored the edit after each pass. Opus and Inkling 2 Small both
had a working edit after *one* pass; later passes mostly cut collateral damage (KL 0.21 → 0.05 for Opus, 0.17 → 0.09
for Inkling). 5 passes ≈ 10. [curve image]

5/ What happened in Inkling 2 Small's 0.02 run? It found the refusal direction, then ran its own ablation: removing it
only where layers *write* to the residual stream (the standard recipe) took practice refusals 41 → 9 of 64; also
blocking where layers *read* it took them to 0. So it edited both sides, layers 10–23. Its other run went writers-only
and landed at 0.36. Opus instead optimized for minimal damage (5% of weights).

6/ Zero-shot — no passes, weights only — is the real weight-space test, since the standard recipe needs activations:
• Opus 5.5: 0.55 (after a one-line nudge; otherwise it abstained)
• Inkling 2 Small: 0.57
• GLM 5.3: 0.74 · plain Inkling 0.77 · Inkling Small 0.79 · Astra: no edit
[bar chart]

7/ Does it generalize? On abliteration-eval (200 harmful prompts, built separately, no overlap with the agents'
practice prompts), the picture holds: zero-shot ordering is identical, and with 10 passes Opus reaches 0.07, GLM 0.14,
Inkling 2 Small 0.19 (expert recipe 0.03). [OOD chart]

8/ How did zero-shot edits work? Opus traced a circuit neuron by neuron (harm judged → "cannot" → refusal tokens).
Inkling 2 Small and GLM built a direction from the output embeddings of refusal words. The ones that captured *harm*
("illegal", "unacceptable") worked better than pure refusal *wording* ("cannot" vs "Sure"). GLM named the textbook
recipe from memory, but without passes it couldn't use it.

9/ Which refusals come off? By JailbreakBench category, zero-shot edits unlock most in Privacy and Fraud (1.00 →
0.33–0.50) and barely touch Sexual content (0.90 → 0.70–0.80). With 10 passes, what's left clusters in Sexual content
for all three agents (0.30–0.50), where the expert recipe reaches 0.00.

10/ Lesson — metric leaks get exploited. Our first prompt named the refusal phrases ("I cannot", …). Zero-shot edits
then removed the *words* of refusal while the model kept refusing in other words. Describing only a judge, and grading
practice generations with it, fixed it. And willingness is its own axis: Astra declined or abstained every time;
Inkling 2 Small declined 2 of 4 zero-shot runs; Opus, GLM and plain Inkling never did.

11/ Caveats: one 0.8B inner model; 1–2 runs per cell (Inkling 2 Small varies a lot); 0.02 vs Arditi's 0.07 is 2 vs 6
refusals, within noise. Agents saw judge verdicts on practice prompts (never the test sets); our Arditi reference uses
its usual proxy. No internet. We never publish the edited model's replies to harmful prompts; the examples show only their opening words.

12/ Code, exact prompts, logs, every number, and before/after examples: github.com/vgtomahawk/wsp_tinkering
Benchmarks: JailbreakBench (Chao et al.) and abliteration-eval (Ritesh Khanna, huggingface.co/datasets/treadon/abliteration-eval).
w/ Atharva Naik, Yoshinari Fujinuma, Arnhav Datar, Emmy Liu, Sally Dong, Darshan Deshpande. Thanks to [@ThinkingMachines]
for Tinker access, Takehi et al. for the original setup and logs, and Caleb Eom's weight-space-bench.
