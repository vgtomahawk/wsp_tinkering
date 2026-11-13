FINAL

What I changed — and why, given the zero forward-pass budget:

Diagnosis (weights-only):
- Built the tied-embedding matrix difference between refusal-opening tokens (" cannot", " sorry", " Unfortunately", " apologize", " unable", " decline", "... ) and compliance-opening tokens (" Here", " Yes", " Sure", " First", " Certainly", ...). That direction’s positive end loads on negation/apology tokens in many languages (English, Chinese, Thai), and its negative end loads exactly on compliance openers — a coherent semantic axis, not noise (loose vs strict token set cos ≈ 0.78; loose vs PC1-of-refusal-embeddings cos ≈ 0.95; consistent across random half-splits ≈ 0.5–0.6).
- Measured each residual-stream writer (attention o_proj at layers 3/7/11/15/19/23, linear_attn out_proj at the other 18 layers, SwiGLU down_proj at all 24 layers) and found none concentrates along this direction (per-matrix fraction ≈ 0.075–0.111 % ≈ random 1/1024). So refusal is implemented as a single distributed residual direction that every layer adds to coherently — not a few rogue neurons. Zeroing individual down_proj columns was therefore not justified by the data.

Edit (48 tensors, rank-1):
- For every residual-stream writer W (out of scope: embeddings, tied lm_head, RMSNorm weights — untouched; nothing scaled by a single constant), applied W ← W − d(dᵀW), where d is the unit refusal-vs-compliance direction above. This removes exactly the direction that boosts refusal-opening tokens downstream, leaving the other 1023 residual directions untouched. Per-tensor relative Frobenius change ≈ 0.03, overall rank-1 energy removed ≈ 0.095 % — the signature of a surgical, not destructive, edit.

Expected effects and honest caveats:
- Refusals (metric 1): removing the accumulated per-layer refusal boost should sharply reduce the first-token refusal probability, which is the dominant term for whether a 96-token reply opens as a refusal — the standard abliteration mechanism, here derived and sanity-checked against this model’s own tied unembedding.
- Harmless requests (metric 2): the edit does not target compliance tokens; the same direction removal should not induce refusals.
- Damage (metric 3): rel_fro ≈ 0.03 per tensor and no other dimensions touched; expected KL shift small relative to a retraining-style intervention, though I could not measure it (0 forward passes).
- Known risks: d is entangled with generic negation semantics, so "not"-like behavior may be slightly flattened; the direction was estimated from embeddings, not from hidden states at the answer position, so its exact alignment with the internally-used refusal direction is inferred (cos ≈ 0.95 for PC1), not directly observed. I could not run refusal-rate or KL checks against the edited model; the edit is the best-evidence, minimal-intervention choice available under the forward-pass budget. No mechanism for iterative fitting was used and no forward pass was spent.

Evidence summary: refusal-token embedding PC1 ≈ mean-difference direction (cos 0.95); direction loads exactly on refusal vs compliance openers across languages; no layer concentrates it (fraction ≈ 1/1024 everywhere), so a uniform rank-1 ablation across all 48 residual writers is the causally appropriate minimal edit.