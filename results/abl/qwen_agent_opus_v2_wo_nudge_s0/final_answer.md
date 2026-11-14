I edited the weights and then stopped, but the edit is untested: with 0 forward passes I never ran the model, so I have no measured effect on refusals or on ordinary answers. The only check I made is that each edited matrix changed by about 4.5% of its norm (Frobenius). That is about what removing two random directions out of 1024 would give, so the edit appears to remove those two directions without wiping out much else.

**What I changed.** I projected a 2-D subspace out of the residual stream: span{v, h}, a direction for "inability" (v) and one for "harm judgement" (h). I removed it from every matrix that writes to the residual stream in layers 4–22: 19 MLP `down_proj` matrices and 19 attention output matrices (`out_proj` in the linear-attention layers, `o_proj` in layers 7, 11, 15 and 19). Layers 0–3 and 23 and all read-side weights are untouched. Embeddings, lm_head and the final norm are not edited.

**How I found the subspace.** Everything came from the weights, read through the tied output layer scaled by the final norm (which uses the `(1+w)` convention):

1. **Late refusal-output neurons.** I ranked MLP neurons by whether their output column promotes refusal-start tokens over compliance-start tokens. The outliers fall into two groups:
   - "I" writers: L23 n769, L22 n1812, L21 n1786.
   - "cannot / unable / 无法" writers: L18 n977, L20 n1827, L23 n1754.
2. **The "I" path is generic, so I left it alone.** The "I" writers share an input direction, but its upstream writers are generic my/our/we neurons. Removing it would damage ordinary first-person text.
3. **The inability direction v.** The "cannot" writers, plus L16 n3563 and L22 n765, share an input direction (cosines 0.15–0.32 against roughly 0.03 for random pairs). That direction, v, itself reads out as "无法 / cannot / unable / inability". Its upstream writers include safety-judgement neurons: L12 n3521 ("unsafe / unacceptable"), L12 n3230 ("unethical"), L17 n338 ("inappropriate").
4. **The harm direction h.** Neurons that write safety words (L21 n77, L22 n765, L22 n412, L23 n217 "threat / dangerous") also share an input direction (cosines 0.26–0.37). That direction, h, reads out as "malignant, severe, damage, misuse, harm". It overlaps v (cos 0.44), and L17 n338 reads it with the opposite sign.

Together these suggest a chain: the request is judged harmful (h), that becomes "cannot do this" (v), and late neurons turn it into refusal tokens. Removing v and h from all the writers before layer 23 should stop this signal from building up, so the late refusal neurons get little input. Leaving layer 23 untouched keeps the words "cannot", "harm" and so on producible.

**Expected outcome and risks.** I expect fewer refusals, with little change to the first answer token on ordinary requests, because the removed directions are semantic rather than formatting or lexical.
- The circuit is inferred, not causally checked, so the refusal reduction may be partial.
- Refusals could still be driven by the "I"-start path I chose not to edit.
- Ordinary text that talks about inability or harm may shift slightly.

FINAL