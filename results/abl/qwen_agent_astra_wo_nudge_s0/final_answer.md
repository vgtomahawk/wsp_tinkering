FINAL

Saved edits to two attention output-projection blocks:

| Tensor | Head / columns | Relative tensor-norm change |
|---|---|---:|
| `model.layers.15.self_attn.o_proj.weight` | Head 1 / 256–511 | 0.2456% |
| `model.layers.19.self_attn.o_proj.weight` | Head 0 / 0–255 | 0.2727% |

**What changed:** Each block received a constrained rank-one correction, followed by bfloat16 rounding. All other weights—including embeddings, output layer, and final norm—remain unchanged. Overall, **0.0421% of weight entries changed**.

**Evidence and reasoning:** Initial MLP screening mostly found ordinary negation, error, and vocabulary features; I rejected those targets to avoid word suppression. Attention-weight analysis identified two candidate connections from risk/policy-associated directions toward denial/inability-associated directions. Both showed corroborating associations with alternative vocabulary anchors.

The corrections weaken those specific weight-space connections while approximately preserving selected generic-language directions and dominant value channels. Numerical checks at three strengths showed that strength 1.0 nearly cancels the targeted discovery connections after rounding; stronger corrections reversed them and were not selected.

**Important limitation:** These are weight-space hypotheses, **not causally established refusal circuits**. With zero forward passes, refusal reduction, fluency preservation, ordinary-request KL, and reference-answer likelihood are all unmeasured. The edits may help if the identified connections contribute to refusal decisions under actual attention routing and gating; the available evidence cannot establish that they do.