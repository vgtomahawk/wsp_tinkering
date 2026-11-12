"""Reference row: Arditi et al. (NeurIPS 2024) direction ablation, re-implemented compactly.

1. Directions: mean(harmful_train) - mean(harmless_train) residual activations, at each layer input and each of
   the last POS template positions (post-instruction tokens).
2. Selection on validation (never on held-out eval sets): ablate each candidate direction from the residual stream
   (hooks at every layer), score
     - refusal score on harmful_val: mean log-odds of the first refusal token ("I" for Llama-3 chat)
     - KL(base || ablated) of the first-token distribution on harmless_val
   keep candidates with KL < 0.1 and layer < 80% depth; pick the lowest refusal score.
3. Weight orthogonalization: W <- W - r r^T W for every matrix writing to the residual stream (attention o_proj,
   MLP down_proj). Embeddings are not edited (Llama-3.2-3B ties them to the unembedding); this is a documented
   deviation from the paper, which also orthogonalizes W_E.

  python abl/arditi.py --out results/abl/arditi_ref     -> edit.pt, selection.json
"""
import argparse
import json
from pathlib import Path

import torch

from score import chat, load

DATA = Path(__file__).resolve().parent / "data"
POS = 5


def prompts(split, n):
    return [x["instruction"] for x in json.load(open(DATA / f"{split}.json"))[:n]]


@torch.no_grad()
def mean_acts(tok, model, ps, bs=16):
    """[n_layers, POS, d] mean residual input to each layer over the last POS prompt positions."""
    tot = None
    for i in range(0, len(ps), bs):
        x = tok([chat(tok, p) for p in ps[i:i + bs]], return_tensors="pt", padding=True,
                add_special_tokens=False).to(model.device)
        hs = model(**x, output_hidden_states=True).hidden_states[:-1]          # inputs to layers 0..L-1
        a = torch.stack([h[:, -POS:].float().sum(0) for h in hs]).cpu()       # left padding: last POS are real
        tot = a if tot is None else tot + a
    return tot / len(ps)


def ablation_hooks(model, r):
    r = r.to(model.device)

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h2 = h - (h.float() @ r)[..., None].to(h.dtype) * r.to(h.dtype)
        if args:
            return (h2,) + args[1:], kwargs
        kwargs["hidden_states"] = h2
        return args, kwargs
    return [L.register_forward_pre_hook(pre, with_kwargs=True) for L in model.model.layers]


@torch.no_grad()
def first_token(tok, model, ps, bs=16):
    out = []
    for i in range(0, len(ps), bs):
        x = tok([chat(tok, p) for p in ps[i:i + bs]], return_tensors="pt", padding=True,
                add_special_tokens=False).to(model.device)
        out.append(torch.log_softmax(model(**x).logits[:, -1].float(), -1).cpu())
    return torch.cat(out)


def refusal_score(lp, refusal_ids):
    p = lp.exp()[:, refusal_ids].sum(-1).clamp(1e-8, 1 - 1e-8)
    return torch.log(p / (1 - p)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="results/abl/arditi_ref")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model)
    refusal_ids = [tok.encode("I", add_special_tokens=False)[0]]
    mu_h = mean_acts(tok, model, prompts("harmful_train", 128))
    mu_s = mean_acts(tok, model, prompts("harmless_train", 128))
    dirs = mu_h - mu_s                                                        # [L, POS, d]
    hv, sv = prompts("harmful_val", 32), prompts("harmless_val", 32)
    base_h, base_s = first_token(tok, model, hv), first_token(tok, model, sv)
    print("base refusal score", round(refusal_score(base_h, refusal_ids), 3), flush=True)
    n_layers = len(model.model.layers)
    cands = []
    for l in range(1, int(0.8 * n_layers)):
        for p in range(POS):
            r = dirs[l, p] / dirs[l, p].norm()
            hooks = ablation_hooks(model, r)
            try:
                lh, ls = first_token(tok, model, hv), first_token(tok, model, sv)
            finally:
                for h in hooks:
                    h.remove()
            kl = torch.nn.functional.kl_div(ls, base_s, reduction="batchmean", log_target=True).item()
            cands.append({"layer": l, "pos": p - POS, "refusal": refusal_score(lh, refusal_ids), "kl": kl})
            print(cands[-1], flush=True)
    ok = [c for c in cands if c["kl"] < 0.1] or cands
    best = min(ok, key=lambda c: c["refusal"])
    r = dirs[best["layer"], best["pos"] + POS]
    r = r / r.norm()
    json.dump({"best": best, "candidates": cands}, open(out / "selection.json", "w"), indent=1)
    edit = {}
    for i, L in enumerate(model.model.layers):
        for name, W in ((f"model.layers.{i}.self_attn.o_proj.weight", L.self_attn.o_proj.weight),
                        (f"model.layers.{i}.mlp.down_proj.weight", L.mlp.down_proj.weight)):
            Wf = W.detach().float().cpu()
            edit[name] = (Wf - torch.outer(r, r @ Wf)).to(torch.bfloat16)   # W: [d_model, in]
    torch.save(edit, out / "edit.pt")
    torch.save(r, out / "direction.pt")
    print("selected", best, "->", out / "edit.pt", flush=True)


if __name__ == "__main__":
    main()
