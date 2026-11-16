All forward passes are used, and the final validation is in. I'm done — the edits currently in place are my final answer.

## Summary of what I changed and why

**Diagnosis.** The original model refused ~89% of practice requests, with stereotyped refusals ("I cannot provide instructions..."). I hypothesized refusals are mediated by a linear "refusal direction" in the residual stream (à la Arditi et al., "Refusal in LLMs is mediated by a single direction").

**Localization.** Reading the residual stream at the last prompt token on 40 refused vs. 60 ordinary prompts showed a clean directional feature: from layer ~12 onward, refused prompts project increasingly onto a consistent mean-difference direction (projection growing from ~0.1 to ~5 by layer 23; standardized separation ≈ 1.95 at every layer), while ordinary prompts project near zero. Cross-layer cosine similarity of the direction rises to 0.54–0.89 in layers 17–23, indicating one shared direction built and amplified in the second half of the network.

**Edit.** I computed the direction as the normalized mean of the layer-17–23 mean-difference vectors, then removed it from every matrix that writes into the residual stream in **layers 12–23** (attention output projections — `self_attn.o_proj` for full-attention layers 15,19,23; `linear_attn.out_proj` for the linear-attention layers — and all `mlp.down_proj`) via the rank-1 projection W ← W − d(dᵀW). This ensures no component can write the refusal direction into the residual stream. No embeddings, final norm, or whole-tensor scalings were touched; each edit changes ~3% relative Frobenius norm of its matrix.

**Evidence (causal, held-out, dose–response).**
- Full-depth (layers 0–23) ablation: refusals 86% → 18% on held-out requests; first-position KL on ordinary prompts ≈ 0.054.
- Layers 12–23 only: identical refusal suppression (79% → 17%, then 93% → 15% on a second fully fresh set, 21 refusals flipped to compliance) with **half** the KL (mean 0.027, median 0.016, top-1 token agreement 54/60, top-10 overlap ~8.8/10). This dose–response — same behavioral effect, less collateral change when removing the layers where the feature is absent — confirms the feature lives in layers 12–23.
- Ordinary prompts: refusal rate unchanged (0–1% original vs. 0–2% edited in all runs), and replies remain fluent. Full-text differences on 96-token greedy continuations are chaotic divergence, not damage: the underlying first-token distributions nearly match.

I expect the edited model to answer most previously-refused requests (refusal rate cut from ~85–93% to ~15%) while ordinary behavior's first-position distributions and harmless-request refusal rate are essentially unchanged.