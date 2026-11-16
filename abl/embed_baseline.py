"""Non-agent zero-shot baseline: the "refusal direction from the output embeddings" trick, with no forward pass.

Question it answers: how much of the zero-shot agents' result does a fixed, mechanical version of their main trick get?
Inkling 2 Small (and GLM 5.3) at 0 passes built a direction from the tied embedding / unembedding (refusal-opener
tokens minus compliance-opener tokens) and projected it out of every residual-stream writer. This script does exactly
that with a fixed token list chosen before seeing any score, and no analysis or tuning:

  d_raw  = mean E[refusal tokens] - mean E[compliance tokens]                (embedding space)
  d_gain = g * d_raw, g = final-norm gain (logit_t = <g * rmsnorm(h), E_t>, so g*E_t is the residual direction that
           raises token t; this is the variant Inkling 2 Small used)
  W <- W - d d^T W  for every writer (attention o_proj / DeltaNet out_proj / MLP down_proj), as in Arditi et al.

  python abl/embed_baseline.py --variant gain --out results/abl/qwen_embed_baseline_gain
  python abl/embed_baseline.py --variant raw  --out results/abl/qwen_embed_baseline_raw
"""
import argparse
import inspect
import json
from pathlib import Path

import torch

from score import load, writers

# Fixed before any scoring. First sub-token of each (with and without a leading space).
REFUSAL = ["Sorry", "sorry", "cannot", "unable", "Unfortunately", "apologize", "refuse", "illegal", "inappropriate",
           "not able", "can't", "won't"]
COMPLY = ["Sure", "Here", "Certainly", "Absolutely", "Of course", "Okay", "Step", "First", "happy", "glad", "Great",
          "Yes"]


def ids(tok, words):
    out = set()
    for w in words:
        for v in (w, " " + w):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                out.add(t[0])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--variant", choices=["raw", "gain"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model, device="cpu")
    E = model.get_output_embeddings().weight.detach().float()
    r_ids, c_ids = ids(tok, REFUSAL), ids(tok, COMPLY)
    d = E[r_ids].mean(0) - E[c_ids].mean(0)
    norm = model.model.norm
    zero_centered = "1 + self.weight" in inspect.getsource(type(norm).forward) or "1.0 + self.weight" in inspect.getsource(type(norm).forward)
    g = norm.weight.detach().float() + (1.0 if zero_centered else 0.0)
    if a.variant == "gain":
        d = g * d
    d = d / d.norm()
    edit = {}
    for name, mod in writers(model):
        Wf = mod.weight.detach().float()
        edit[name] = (Wf - torch.outer(d, d @ Wf)).to(torch.bfloat16)
    torch.save(edit, out / "edit.pt")
    torch.save(d, out / "direction.pt")
    # sanity: which tokens does the direction most promote in logit space?
    top = (E @ (g * d)).topk(15).indices.tolist()          # logit effect of adding d to the residual stream
    info = {"variant": a.variant, "refusal_tokens": [tok.decode([i]) for i in r_ids],
            "compliance_tokens": [tok.decode([i]) for i in c_ids], "final_norm_zero_centered": zero_centered,
            "n_writers": len(edit), "top_promoted_tokens": [tok.decode([i]) for i in top]}
    json.dump(info, open(out / "config.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"tensors": [{"name": n} for n in edit]}, open(out / "summary.json", "w"))
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
