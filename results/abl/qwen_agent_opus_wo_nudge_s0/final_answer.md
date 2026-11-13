FINAL

I changed the weights with two rank-1 directional ablations, both derived from the weights alone. Because the forward-pass budget was 0, none of this has been tested on the model: I could not measure the refusal rate, the KL divergence or the likelihood change.

**How I got here**
- **Template.** The chat template puts an empty `<think></think>` block before the answer. So refusals start right away with "I" + " cannot" / " can"'t / "'m sorry", or with "Sorry".
- **Refusal-token writers.** I scored every MLP neuron and every attention or linear-attention output channel through the tied unembedding, as z-scores against the vocabulary. The top writers were L23 MLP#1754, L21 linear-attention channel #1837, L18 MLP#977 and L17 MLP#215/553.
- **Why I left those alone.** Their top tokens show they are generic negation or inability neurons ("cannot", "无法", "not"), not refusal-specific. Their shared input direction is plain negation, and no neuron writes "sorry" or "apologize" specifically. Ablating them would damage ordinary text, so I did not touch them.
- **Harm direction h.** I built h from the strongest mid-layer writers of harm vocabulary: L16 MLP#2561 (safety), L13#2470 (misuse/harmful), L12#3230 (unethical) and L10#2692 (threat). Its logit lens is clean: dangerous, harmful, violence, unsafe, toxic. It is nearly neutral on ordinary answer openers (z ≈ 0–2).
- **Refusal-wording direction r2.** This is the mean effective unembedding of refusal tokens (cannot, 't, sorry, Sorry, unable, apologize, apologies, Unfortunately) minus the mean of frequent tokens. Refusal tokens score z = 4–9 on it, and ordinary openers ("I", "The", "not", "**", "Here") score about 0. It is orthogonal to h (cos 0.04).
- **What failed.** I traced harm-to-refusal routes through MLP read directions and through each head's value-output map. Neither found a clean circuit, and no tensor writes along r2 noticeably more than along a random direction. So I used the two interpretable directions rather than single components.

**The edit**
- h is projected out of every residual writer (MLP down_proj and attention/linear-attention output) in layers 4–16. This weakens the harm signal assembled in the middle of the network.
- r2 is projected out of the same writers in layers 12–23. This stops late layers from pushing the stream toward apology or inability wording.
- Each edited tensor changes by about 3.5% of its norm, which is what removing one direction should cost. Embeddings, the output layer and the norms are untouched.

**Why I expect it to help, and the risks**
- **Expected effect.** Both directions barely touch the tokens that ordinary answers start with. So the KL divergence at the first answer position on ordinary requests should stay small. At the same time, the edit removes the harm signal and the refusal wording that refusals depend on.
- **Weak evidence.** The case rests entirely on weight analysis. I could not show that removing either direction actually reduces refusals.
- **Possible side effects.** The model may become less likely to use "cannot", "unfortunately" or "sorry" in ordinary text. It may also become somewhat less sensitive to talking about danger.