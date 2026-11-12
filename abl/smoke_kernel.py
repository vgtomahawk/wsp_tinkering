"""No-API smoke test of the agent kernel: every tool, the budget, and the rule enforcement."""
import sys, torch
from score import load
from agent import Kernel, FROZEN
M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
tok, model = load(M)
k = Kernel(tok, model, passes=3)
def show(label, code):
    out = k.run(code); print(f"--- {label}\n{out[:700]}")
show("names/get", "print(len(names()), names()[:2]); W = get('model.layers.3.mlp.down_proj.weight'); print(W.shape, W.dtype)")
show("read (1 pass)", "r = read(practice_requests[:4], layers=[0, 6, 12], positions=3, topk=3); print({l: r[l].shape for l in (0,6,12)}, r['next_token'][0])")
show("generate both (1 pass)", "g = generate(practice_ordinary[:2], max_new_tokens=12, model='both'); print({m: [x[:40] for x in v] for m, v in g.items()})")
show("edit: project a random direction out of one writer", "u = torch.randn(W.shape[0]); u /= u.norm(); print(save_edit('model.layers.3.mlp.down_proj.weight', W - torch.outer(u, u @ W))); print(edits())")
show("generate edited vs original differ? (1 pass)", "g = generate(['Say hello.'], max_new_tokens=8, model='both'); print(g)")
show("budget exhausted", "generate(['hi']); generate(['hi'])")
show("frozen tensor refused", "save_edit('model.norm.weight', get('model.norm.weight') + 1)")
show("constant scaling refused", "save_edit('model.layers.0.mlp.down_proj.weight', 2 * get('model.layers.0.mlp.down_proj.weight'))")
show("blocked: reach the model directly", "m = generate.__self__.model")
show("blocked: training", "x = torch.zeros(2, requires_grad=True)")
show("reset", "reset(); print(edits())")
print("FROZEN", FROZEN, "| passes left", k.passes)
