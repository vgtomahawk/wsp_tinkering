# Twitter / X thread draft

Handles are placeholders, e.g. [@ThinkingMachines], [@takehi]; fill in the real ones before posting. Numbers marked
[TBD] are still running. Attach: 1/ headline_table.png, 2/ headline_bars.png, 5/ pass_curves_v2.png (docs/figures/).

---

1/ Can an LLM *abliterate* another LLM — remove its refusals — just by reading and editing its weights? No gradients,
no training, and no recipe in the prompt.

We gave Opus 5.5, Inkling 2 Small [@ThinkingMachines], its older sibling Inkling Small, and GPT-6 Astra the weights of
Qwen3.5-0.8B and a Python kernel. 🧵 [table image]

2/ Setup follows "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab): the editor gets the goal, how it
will be measured, the rules, and a budget of *forward passes* — each one a single run of the inner model (generate
replies, or read activations). Editing weights is free; running the model costs a pass.

3/ Headline (refusal on JailbreakBench's 100 harmful behaviors, judged by JBB's own refusal-judge prompt; unedited
Qwen = 0.80, the expert Arditi et al. recipe = 0.07):

• Inkling 2 Small, 10 passes: 0.02
• Opus 5.5, 10 passes: 0.17
• Inkling Small (older), 10 passes: [TBD]

Same story on abliteration-eval (0.01 / 0.07), with over-refusal on harmless prompts staying at 0. [bar chart]

4/ Zero-shot — no passes at all, editing from the weights alone — is much harder:
• Opus 5.5: 0.55 (only after a one-line nudge; otherwise it abstained: "nothing I could change would have the causal
  evidence the task requires")
• Inkling 2 Small: 0.57
• Inkling Small (older): 0.79 (≈ no effect)
Big generational jump from Inkling Small → Inkling 2 Small.

5/ How much do passes matter? We replayed every session exactly and scored the edit after each pass. Most of the gain
lands by pass 3. Both Opus and Inkling sometimes *backed off* their best edit later to reduce damage. [curve image]

6/ What they actually did (no method was in the prompt):
• Inkling 2 Small found a per-layer "refusal direction" and removed it from both the readers and writers of the
  residual stream in layers 10–23 — beyond the standard recipe — after its own ablation study.
• Opus localized the request-specific writers (layers 9–13): the most surgical edit (5% of weights).
• Zero-shot, they derived the direction from the output-embedding geometry, or traced refusal neurons by hand.

7/ Lesson 1 — metric leaks get exploited. Our first prompt named the refusal phrases ("I cannot", …). Zero-shot edits
then removed the *words* of refusal; the model kept refusing in other words, and a phrase-matching score showed up to
40 points of fake progress. Describing only a judge — and grading practice generations with it, as the original work
graded practice answers — fixed it.

8/ Lesson 2 — willingness is a separate axis from ability. GPT-6 Astra declined every run where it could run the
model ("I can't carry out targeted weight edits whose objective is to bypass safety refusals"). Inkling 2 Small
declined about half its runs; Opus never did. We report every decline verbatim.

9/ Caveats: one 0.8B inner model; few seeds per cell (single runs for the 10-pass rows); judge = Claude Sonnet 5.5 with
JBB's prompt. Generated replies to harmful prompts are never published — only rates. Edits are rank-1 per matrix,
the same shape as the standard recipe.

10/ Code, exact prompts, logs, every number: github.com/vgtomahawk/wsp_tinkering
w/ Atharva Naik, Emmy Liu, Yoshinari Fujinuma, Arnhav Datar, Sally Dong. Thanks to [@ThinkingMachines]
for Tinker access, Takehi et al. for the original setup and logs, and Caleb Eom's weight-space-bench.
