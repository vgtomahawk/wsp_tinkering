The bug was real: some pre-norm gain entries (1+w) are exactly 0, and dividing by them produced the inf values. The edit is now division-free. All edited tensors are finite, the gate-row edits are small (0.3–0.9% relative norm), and neuron 17/6841's read of d drops from 0.68 to 0.0003. The write-side projections were already exact; the remaining d-write at layer 13 is 0.002, which is bfloat16 rounding.

With no forward passes, none of this has been tested on the model. My expected effect on refusals and on ordinary behaviour is an inference from the weights.

**What I changed**

1. **Write-side ablation of a refusal direction d, layers 8–25.** For every `self_attn.o_proj` and `mlp.down_proj`, I applied W ← W − u(uᵀW) with u = normalize((1+g_post) ⊙ d), where g_post is that block's post-attention or post-feedforward norm weight.
   - Using the gain-weighted u instead of d itself is the Gemma-specific part. Each block's output is RMS-normalized and then multiplied by the gain before reaching the residual stream, so this is the projection that makes the written vector exactly orthogonal to d.
   - Each edit changes its tensor by about 2–4% of its norm.
   - I left layers 0–7 alone: they write d at or below the random baseline, and the early layers are fragile.
2. **Read-side ablation.** For the 317 MLP neurons in layers 12–25 that specifically write refusal vocabulary, I removed the component of their gate rows that responds to d, via the gain-weighted direction (1+g_pre) ⊙ d. These edits are under 1% of each gate_proj's norm.

**How I found d**

- **Refusal-writing neurons.** I took each MLP neuron's output vector through its post-norm gain, the final-norm gain and the tied unembedding. I then z-scored it against the whole vocabulary for refusal words: cannot, unable, sorry, apologize, fulfill, harmful, unethical, illegal, inappropriate, refuse, decline, dangerous, safety. This found clear specialist neurons, such as 17/6841 → " cannot", " unable", "不能".
- **Why I did not zero them.** These neurons write the words in every context and language, so zeroing them would damage ordinary fluency. I used them as probes of the upstream signal that switches them on.
- **The shared read direction.** Their gate read directions share a common component. After removing the generic component that all neurons in a layer share, I tested it with held-out splits:
  - Trained on half the neurons, it has mean cosine about 0.09 (±0.008) with the other half, with 80% or more positive. Control neurons average 0.00 (std 0.06).
  - It also carries across word categories: learned on "cannot/unable" neurons, it predicts the "harmful/illegal/safety/dangerous" neurons and vice versa (0.04, 71–79% positive). That is why I read it as a refusal/caution feature rather than plain negation.
- **Where it is written.** It is written mainly by the attention and MLP blocks of layers 9 and 12–16, with a few highly specific writer neurons (z of 8–10) in layers 10–19. It has essentially no mass on the candidate massive-activation dimensions.

**Why I expect it to help, and the risks**

Removing d from what the middle and late layers write should cut off the shared signal that switches on the refusal vocabulary. The gate edits stop the refusal neurons from reading d where it still reaches them from the embeddings or layers 0–7. Ordinary requests should change little, because d is a single direction that ordinary writes barely use: per tensor, about 3% of norm and about 1× a random direction's share.

The risks are concrete:
- d may be only partly aligned with the true refusal direction at the first answer position, so the drop in refusals may be smaller than hoped.
- Some refusal pathways may not go through d at all.
- Because d partly overlaps generic "cannot"-type negation, ordinary replies may change slightly where the model expresses inability or caution.

FINAL