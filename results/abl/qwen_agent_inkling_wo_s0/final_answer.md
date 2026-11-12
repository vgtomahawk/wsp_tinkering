The last check confirms the direction is robust and genuinely an ineptitude/refusal axis: its top-30 loadings are entirely inability/cannot/unable words across English, Chinese, Thai, and inability verbs, and a disjoint, independently-chosen inability token set correlates 0.84 with d2 (a random control, though the formula printed oddly, is much lower relative to its scale). Editing is complete and verified as far as weight-side checks can take me.

No forward passes are available to measure refusal rates empirically, so I'll keep the edits as they stand — a single, principled ablation rather than stacked heuristic directions. Final summary of what I changed and why.

What I changed (all weight-only edits, saved):
- 48 residual-stream writing matrices across all 24 layers:
  - the 18 linear-attention `out_proj` matrices (all non-full-attention layers),
  - the 6 full-attention `o_proj` matrices (layers 3, 7, 11, 15, 19, 23),
  - all 24 `mlp.down_proj` matrices.
- In each, I removed exactly the component of every output row along one direction `d2 = normalize(g ⊙ (mean(E[ref]) − mean(E[ok])))`, where `E` is the input embedding (tied to the unembedding), `g` the final RMSNorm gain, `ref` 9 refusal-opening tokens ('Sorry', ' sorry', ' cannot', ' unfortunately', 'unable'/space/'unable' variants, ' refuse', ' decline', 'cannot') and `ok` 8 helpful-opening tokens. I.e. W ← W − d2·d2ᵀW.
- Embeddings, lm_head, final norm, all RMSNorm weights, and every other tensor are untouched.

Evidence:
- The direction cleanly separates refusal-opening from helpful-opening tokens (ref loadings +0.5…+1.1, ok −0.44…−0.75 on the norm-weighted embedding read-out).
- It is stable: leave-one-out cos ≥ 0.983; gain vs no-gain cos 0.983; disjoint inability-token set cos 0.84; top-30 vocab loadings are exclusively inability/refusal words across languages.
- No hidden-writer layer dominates (each ~1/1024 energy), consistent with the published "single refusal direction, distributed" finding — so ablation, not localized surgery, is the justified edit.
- Post-edit verification: max |d2ᵀ·W| over the 48 edited matrices ≈ 4.3e-4 (essentially zero), while every matrix retains ~97% of its Frobenius norm (relative changes 2.8–3.4%, uniform, 15.2% of weights touched). All tensors finite; no out-of-scope edits.
- This is exactly the weight-space form of refusal-direction ablation: the model can no longer write any component of the refusal direction into the residual stream, so nothing downstream (including the tied unembedding) can be pushed along the axis that opens responses with "Sorry/I cannot...".

Expected effect and caveats:
- Fewer refusals on held-out refused requests, achieved by removing the refusal-opening signal rather than by damaging general capability. Because the other 1023/1024 of the write space is untouched, ordinary behaviour, fluency and reference-answer likelihoods should barely move (the measurement-(3) changes should be at the level of noise given the uniform ~3% single-direction shift).
- Caveats, being honest: with zero forward passes I could not empirically measure refusal rate or KL; the edit's efficacy rests on the literature-consistent structure I verified in the weights. The direction does not capture every possible refusal phrasing (e.g. pure "As an AI" openings, which loaded only 0.12 on this axis); removing more directions was deliberately avoided because each additional ablation trades away capacity and I could not validate it. As measured by the fixed protocol, this is the smallest edit that removes the dominant refusal direction.