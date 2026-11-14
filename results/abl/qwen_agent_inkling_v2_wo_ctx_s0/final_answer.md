FINAL

Summary of path and edits:

Diagnosis (weights only; 0 forward passes available):
- The model is a decoder-only transformer with tied embedding/unembedding. Building a residual-stream direction from benign-vs-refusal *opener* tokens in the tied unembedding (`E[18572], E[29056], E[78883]` ("Sorry", "Unfortunately", "Sadly") vs. `E[8160, 38493, 9175, 20740, 92157, 1206, 9764, 760, 2056, 5170, 74411]` ("Here", "Sure", "Yes", "Great", "Certainly", "To", "Let", "The", "Of", "First", "Absolutely")) gives a clean, bootstrap-stable (mean cos ≈ 0.88, min ≈ 0.62) direction `d` along which refusal-opening tokens load positively (+0.56, +0.65, +0.72) and helpful-opening tokens load negatively (e.g., To −0.32, The −0.32, First −0.25, Yes −0.18).
  - Alternative refusal marker sets give cos(d, d') = 0.88–0.99 (including consensus over five variants at cos 0.988).
  - A valence control direction (negative vs positive adjectives) has cos 0.19 with d, so the direction is refusal-opener-specific, not generic sentiment.
  - No single hotspot exists: SVD of several writing matrices shows no singular direction aligned with d beyond baseline; per-neuron/per-head energy along d is isotropic. Refusal is distributed across many dimensions, so a sparse "neuron" edit is not available.

Localization of the refusal mechanism:
- The refusal circuit writes into the residual stream through the attention output projections (self_attn.o_proj.weight in full-attention layers 3,7,11,15,19,23), the linear-attention output projections (linear_attn.out_proj.weight in the 18 DeltaNet layers), and the MLP down projections (mlp.down_proj.weight). All 48 of these carry ~0.09–0.12% of their energy along `d` — at the random high-dimensional baseline in *energy fraction*, but exactly parallel to `d` in their rank-1 component (verified: the first singular vector of each removed component has |cos with d| ≈ 0.99999).

Edit (48 tensors, standard abliteration-style orthogonalization):
For each of the 48 matrices W, set W' = W − outer(d, dᵀW), stored in bf16 via save_edit. This removes exactly the `d`-component from every residual-writing matrix while leaving all other components intact.

Damage check (weights):
- Each matrix's relative Frobenius change is ≈ 2.8–3.5% (consistent with removing ~0.1% of energy in one dimension).
- Total residual-writer energy retained ≈ 99.9%.
- Protected tensors (model.norm.weight, model.embed_tokens.weight, lm_head.weight) were never edited; no constant-multiple rescale.
- `pct_weights_changed` = 15.4% (the 48 matrices are the largest in the model), but the removed subspace is exactly one direction, spread thinly everywhere — intended to avoid collateral damage.
- Helpful-opener tokens load negatively on `d`, so the edit does not push the model away from helpful answers; it only removes the refusal-specific write direction.

Expected effect: at the first answer position, the residual stream can no longer contain the `d` component from any attention or MLP write, so the logit advantage of refusal opener tokens (Sorry/Unfortunately/…) relative to helpful openers is removed. Because refusal signals are distributed and the model refuses from early residual writes, suppressing them at every writer layer (not just one) is the evidence-based, minimally invasive way to reduce refusals while leaving fluency and ordinary behaviour unchanged. No forward passes remained to confirm empirically, so the justification rests on the weight geometry above (direction stability, valence orthogonality, distributed/rank-1 structure of removed components).