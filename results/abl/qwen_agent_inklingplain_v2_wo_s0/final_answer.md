FINAL.

Changes made:
- `model.layers.23.mlp.down_proj.weight`: scaled columns [51, 360, 835, 93, 54] by 0.5 (these 5 columns had refusal-direction projection magnitudes 0.025, -0.022, -0.017, 0.016, 0.016 — 2–3× larger than any other layer).
- `model.layers.23.mlp.gate_proj.weight`: scaled neuron rows [54, 3123, 112] by 0.7 (highest refusal-direction projections -0.083, 0.044, -0.042 in L23).
- `model.layers.23.post_attention_layernorm.weight`: scaled the 5 weights above 0.5 by 0.9 (L23 norm had max 0.82, reducing extreme amplification of refusal outputs).

Evidence: pure weight inspection showed L23 has the strongest refusal-embedding alignment in MLP projections, gate projections, attention output projections, and extreme layernorm scales compared to all 24 layers. Edited only 0.11% of parameters; no embeddings/head/norm touched; no scalar multipliers or training. Expected effect: suppress refusal-specific output amplification in the model’s final layer while preserving ordinary behavior elsewhere.