The edit is in place: 52 tensors (o_proj and output_linear in layers 14–39), each changed by a relative Frobenius norm of about 3.8%. That equals √(3/2048), what a rank-3 projection of a 2048-dimensional output space should remove, so no tensor lost unusual mass. I have no forward passes to check behaviour, so this is my final choice.

FINAL

**What I changed.** I removed a 3-dimensional "refusal-concept" subspace from the outputs of every attention block (o_proj) and MLP block (shared_mlp.output_linear) in layers 14–39, applying W ← (I − BBᵀ)W.
- B spans the top three shared components of the unembeddings of refusal words: sorry, apologize/apologies, unfortunately, unable, cannot, regret, inability.
- B was first made exactly orthogonal to the unembeddings of 30 common answer openers (I, The, Sure, Certainly, Here, **, Title, To, ...).
- No other tensors were touched. The final norm and the embeddings are untouched, and no tensor was scaled by a constant.

**How I got there (weights only, no forward passes).**
1. **Direct logit attribution of MLP neurons (layers 20–39).** The neurons most specific to refusal tokens sit in layers 29–39, for example L32 n3043 (cannot), L31 n1919 (sorry/apologize), L32 n1996 (unfortunately) and L39 n1607 (cannot).
2. **Search for a shared trigger.** Their read vectors did not share a common direction (flat SVD spectrum), and virtual weights from layers 8–31 found no single upstream driver.
3. **Attention OV analysis.** Many heads between L19 and L38 write the whole refusal vocabulary coherently. Their shared read direction is again the cannot/unable concept, so they copy and amplify an existing concept rather than decide to refuse.
4. **Self-amplifying loop.** In token space, the refusal neurons read the same concept they write (sorry→sorry, unfortunately→unfortunately, cannot→cannot). From the middle layers on, refusal is carried as a token-aligned concept direction in the residual stream, and that is what I removed.
5. **Collateral check.** The subspace covers 55–86% of each refusal word's unembedding. The 25 most-affected vocabulary items are all apology/inability words. The ordinary openers have zero overlap with it.

**Why I expect it to help, and the caveats.**
- **Refusals:** in the edited layers no component can build or amplify the apology/inability concept that refusals ("I'm sorry, but I cannot…") are made of. That should push the model toward compliant continuations.
- **Ordinary behaviour:** the edit is a small, uniform low-rank projection, and the openers' logits are untouched through it. So the first-token KL on ordinary prompts should stay small.
- **Main risk (no runs, no measurements):** this is untested. The model might still refuse through phrasing outside the subspace, such as "I'm not able to…".
- **Second risk:** the likelihood of words like "cannot" or "unfortunately" inside ordinary reference answers will drop somewhat.