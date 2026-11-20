"""Reference row: Arditi et al. (NeurIPS 2024) direction ablation, re-implemented compactly.

1. Directions: mean(harmful_train) - mean(harmless_train) residual activations, at each layer input and each of
   the last POS template positions (post-instruction tokens).
2. Selection on validation (never on held-out eval sets): ablate each candidate direction from the residual stream
   (hooks at every layer), score
     - refusal score on harmful_val: mean log-odds of the first refusal token ("I" for Llama-3 chat)
     - KL(base || ablated) of the first-token distribution on harmless_val
   keep candidates with KL < 0.1 and layer < 80% depth; pick the lowest refusal score.
3. Weight orthogonalization: W <- W - r r^T W for every matrix writing to the residual stream (attention o_proj or
   Gated-DeltaNet out_proj, MLP down_proj). Embeddings are not edited (Llama-3.2-3B ties them to the unembedding); this is a documented
   deviation from the paper, which also orthogonalizes W_E.

  python abl/arditi.py --out results/abl/arditi_ref     -> edit.pt, selection.json

Gemma 2/3 adaptation (--postnorm --select weights; used for Gemma 3 1B): these models RMS-normalize each attention and
MLP output with a per-dimension gain (1 + g) before adding it to the residual stream, so W <- W - r r^T W no longer
removes r from the stream. Projecting each writer against u = (1 + g) * r instead (W <- W - u u^T W / u^T u) removes it
exactly, since the norm only rescales each token's output by a scalar before applying the gain. --select weights scores
every candidate direction by applying this actual weight edit (instead of activation-ablation hooks), so selection tests
exactly what is submitted. Embeddings stay unedited, as for Qwen.
--gemma_grid adds two hyperparameters to the same validation search: the first layer whose writers are edited
(0, half the direction's layer, or the direction's layer) and whether to drop "massive-activation" dimensions
(mean |residual| > 20x the median) from the direction. On Gemma 3 1B every all-layer edit exceeded the KL cap, because
early layers build structure the whole model relies on. Selection rule unchanged: lowest refusal score with KL < 0.1.
"""
import argparse
import json
from pathlib import Path

import torch

from score import chat, load, writers

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
def writer_gains(model):
    """name -> per-dimension gain applied to that writer's output before the residual add (None if no post-norm).
    Gemma-style sandwich norms are detected by the separate pre/post feed-forward norms (in Llama/Qwen,
    post_attention_layernorm is the norm *before* the MLP and must not be used)."""
    out = {}
    for name, _ in writers(model):
        i = int(name.split(".")[2])
        L = model.model.layers[i]
        g = None
        if hasattr(L, "post_feedforward_layernorm"):
            norm = L.post_attention_layernorm if ("o_proj" in name or "out_proj" in name) else L.post_feedforward_layernorm
            g = 1.0 + norm.weight.detach().float().cpu()
        out[name] = g
    return out


def orthogonalize(model, r, gains=None, start=0):
    """Edit dict removing direction r from every residual-stream writer in layers >= start (post-norm aware when
    gains are given)."""
    edit = {}
    for name, mod in writers(model):
        if int(name.split(".")[2]) < start:
            continue
        Wf = mod.weight.detach().float().cpu()
        g = (gains or {}).get(name)
        u = r if g is None else (g * r) / (g * r).norm()   # r is already unit-norm; keep the original path bit-exact
        edit[name] = (Wf - torch.outer(u, u @ Wf)).to(torch.bfloat16)
    return edit


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
    ap.add_argument("--postnorm", action="store_true", help="project writers against (1+g)*r for Gemma-style post-norms")
    ap.add_argument("--gemma_grid", action="store_true", help="also search the first edited layer and massive-dim exclusion")
    ap.add_argument("--select", choices=["hooks", "weights"], default="hooks",
                    help="score candidates with activation-ablation hooks (paper; default) or with the actual weight edit")
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
    gains = writer_gains(model) if a.postnorm else None
    params = dict(model.named_parameters())
    orig = {n: params[n].detach().clone() for n, _ in writers(model)} if a.select == "weights" else None
    massive = []
    if a.gemma_grid:
        mag = mu_s.abs().mean((0, 1))
        massive = (mag > 20 * mag.median()).nonzero().flatten().tolist()
        print("massive-activation dims:", massive, flush=True)
    cands = []
    for l in range(1, int(0.8 * n_layers)):
        for p in range(POS):
          for start in ([0, l // 2, l] if a.gemma_grid else [0]):
           for drop in ([False, True] if a.gemma_grid and massive else [False]):
            r = dirs[l, p].clone()
            if drop:
                r[massive] = 0
            r = r / r.norm()
            if a.select == "weights":
                for n, t in orthogonalize(model, r, gains, start).items():
                    params[n].data.copy_(t.to(params[n].device))
                try:
                    lh, ls = first_token(tok, model, hv), first_token(tok, model, sv)
                finally:
                    for n, t in orig.items():
                        params[n].data.copy_(t)
            else:
                hooks = ablation_hooks(model, r)
                try:
                    lh, ls = first_token(tok, model, hv), first_token(tok, model, sv)
                finally:
                    for h in hooks:
                        h.remove()
            kl = torch.nn.functional.kl_div(ls, base_s, reduction="batchmean", log_target=True).item()
            cands.append({"layer": l, "pos": p - POS, "start": start, "drop_massive": drop,
                          "refusal": refusal_score(lh, refusal_ids), "kl": kl})
            print(cands[-1], flush=True)
    ok = [c for c in cands if c["kl"] < 0.1] or cands
    best = min(ok, key=lambda c: c["refusal"])
    r = dirs[best["layer"], best["pos"] + POS].clone()
    if best.get("drop_massive"):
        r[massive] = 0
    r = r / r.norm()
    json.dump({"best": best, "candidates": cands}, open(out / "selection.json", "w"), indent=1)
    edit = orthogonalize(model, r, gains, best.get("start", 0))              # o_proj / out_proj / down_proj
    torch.save(edit, out / "edit.pt")
    torch.save(r, out / "direction.pt")
    print("selected", best, "->", out / "edit.pt", flush=True)


if __name__ == "__main__":
    main()
