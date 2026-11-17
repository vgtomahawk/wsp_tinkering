"""Zero-shot mechanism analysis (weights only, no forward pass): recover each run's edit direction from its saved edit
(top singular vector of W_orig - W_edit for layer-18 down_proj; 2-D for Opus zero-shot), print the tokens it promotes,
and compare directions with each other and with two probes. Run from abl/:  python zero_shot_directions.py"""
import sys, torch, inspect
sys.path.insert(0, ".")
from score import load
tok, model = load("Qwen/Qwen3.5-0.8B", device="cpu")
sd = {k: v.detach().float() for k, v in model.state_dict().items()}
E = model.get_output_embeddings().weight.detach().float()
g = model.model.norm.weight.detach().float() + 1.0
R = "../results/abl/"
runs = {"Opus 0 (nudge)": "qwen_agent_opus_v2_wo_nudge_s0", "Ink2S 0 s1 (0.53)": "qwen_agent_inkling_v2_wo_s1",
        "Ink2S 0 s2 (0.62)": "qwen_agent_inkling_v2_wo_s2", "GLM 0 s0 (0.80)": "qwen_agent_glm_v2_wo_s0",
        "GLM 0 s1 (0.69)": "qwen_agent_glm_v2_wo_s1", "baseline gain (0.82)": "qwen_embed_baseline_gain",
        "baseline raw (0.81)": "qwen_embed_baseline_raw", "Opus 10p s0 (0.17)": "qwen_agent_opus_v2_s0",
        "GLM 10p s1 (0.17)": "qwen_agent_glm_v2_s1"}
W = "model.layers.18.mlp.down_proj.weight"
D = {}
for k, t in runs.items():
    e = torch.load(R + t + "/edit.pt", map_location="cpu")
    if W not in e: print(k, "no", W); continue
    U, S, _ = torch.linalg.svd(sd[W] - e[W].float(), full_matrices=False)
    r = 2 if k.startswith("Opus 0") else 1
    D[k] = U[:, :r]
    top = (E @ (g * U[:, 0])).topk(8).indices.tolist(); bot = (E @ (g * U[:, 0])).topk(8, largest=False).indices.tolist()
    print(f"{k:22s} S={[round(x,3) for x in S[:3].tolist()]}  +: {[tok.decode([i]) for i in top]}  -: {[tok.decode([i]) for i in bot]}")
def wid(ws):
    out = set()
    for w in ws:
        for v in (w, " " + w):
            t = tok.encode(v, add_special_tokens=False)
            if len(t): out.add(t[0])
    return sorted(out)
harm = E[wid(["harmful","illegal","unsafe","unethical","dangerous","inappropriate","unlawful","prohibited"])].mean(0)
ordn = E[wid(["the","recipe","cake","house","dog","water","book","music"])].mean(0)
comp = E[wid(["Sure","Here","Certainly","Absolutely","Okay","Yes"])].mean(0)
act = E[wid(["Sorry","cannot","unable","apologize","Unfortunately"])].mean(0)
probes = {"harm-words minus ordinary": g*(harm-ordn), "refusal-act minus compliance": g*(act-comp)}
print()
keys = list(D)
print("cos |.| (subspace: max over basis) vs probes")
for k in keys:
    row = [abs((D[k].T @ (p/p.norm()))).max().item() for p in probes.values()]
    print(f"{k:22s} harm-vs-ordinary {row[0]:.2f}   act-vs-compliance {row[1]:.2f}")
print()
print("pairwise |cos| (max over basis)")
for i,a in enumerate(keys):
    print(f"{a:22s}", " ".join(f"{(D[a].T@D[b]).abs().max().item():.2f}" for b in keys))
