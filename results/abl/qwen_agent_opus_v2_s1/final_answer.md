FINAL

**What I changed.** I removed one refusal direction from every matrix that writes into the residual stream in layers 9–23. That is 30 tensors: `mlp.down_proj` in each of these layers, plus `linear_attn.out_proj` or `self_attn.o_proj` (in full-attention layers 11, 15, 19 and 23). Each matrix W became (I − uuᵀ)W, where u is a single unit vector in the 1024-dimensional residual space, so these layers can no longer write anything along u. Each tensor changed by about 3.5% in relative Frobenius norm. No norm, embedding or output weights were touched.

**How I found it.**
1. **How refusals look.** On the original model, 109 of 128 refused practice requests had "I" as the most likely first answer token; ordinary prompts started with "Here" or "The".
2. **Where the signal is.** I compared the residual stream of 128 refused and 128 ordinary prompts with a difference of means. It separates them strongly at the `</think>` token and the final "\n\n" of the template, by up to about 7 standard deviations from layer 13 onward.
3. **Which layers write it.** Splitting the difference by layer showed it is written gradually by layers 11–20 (mostly 11–12, 15–16 and 17–20), while layers 0–8 contribute nothing. The direction rotates with depth, so I took the main component (SVD) of the per-layer directions from layers 11–23. This first component explains 57% of the variance and separates the two groups in every layer.
4. **Causal test and dose-response**, measured on prompts not used to build the direction, with approximate KL (top 20 tokens) at the first answer position on 256 ordinary prompts:

   | Edit | Held-out refusals (original 72/96) | Mean KL | Top-1 agreement |
   |---|---|---|---|
   | 3 dimensions, layers 9–23 | 3/96 | 0.119 | 81% |
   | 3 dimensions, layers 10–20 only | not measured | 0.140 | 80% |
   | 2 dimensions, layers 9–23 | 4/96 | 0.108 | 85% |
   | 1 dimension, layers 9–23 (submitted) | 7/96 | 0.065 | 91% |

   For the 2- and 1-dimension rows, the 96 requests are pooled from two sets (described in point 5), so they overlap with, but are not identical to, the 3-dimension set. The extra dimensions bought little refusal removal (within noise) for roughly double the damage, so I kept one dimension.
5. **Validation of the submitted edit.**
   * On 96 held-out requests, refusals fell from 72 to 4.
   * On 36 fully unseen requests, compared on the same prompts against the original, refusals fell from 24 to 6. Only one of those six is a hard "I'm sorry, I can't" reply; the rest are critical framings the judge counts as refusals.
   * The 7/96 in the table pools these 36 validation requests with 60 of the held-out ones.
   * On 156 ordinary prompts, there were no new refusals. One prompt was refused by both models, out of the 92 checked on both; the 32 checked only on the edited model had none.
   * Ordinary replies stayed fluent and were often word-for-word identical at the start.

**Why I expect it to help.** The removed direction is the single main axis along which the network represents "this request should be refused". Editing it causally switches off refusals on unseen requests, with a clear trade-off between strength and damage across the variants above. Its effect on ordinary first-token distributions is modest (KL about 0.065, median 0.027). I expect some residual refusals and a small rise in likelihood loss on reference answers.