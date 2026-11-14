"""Rebuild the edit an agent had in place after each forward pass, by replaying its logged code cells.

The agent's code cells (log.jsonl) are re-run in order in a fresh kernel over the same inner model; forward passes are
greedy, so they reproduce the original outputs up to small MPS numerical differences. After every replayed cell we
record the passes used; the snapshot for budget k is the edit in place at the end of the last cell that had used at
most k passes, i.e. what the agent would have submitted had it been stopped there. Only distinct edit states are
saved. A check at the end compares the replayed final edit with the run's real edit.pt.

  python abl/replay.py --run results/abl/qwen_agent_inkling_p10_ctx_s1 --inner Qwen/Qwen3.5-0.8B
  -> <run>/replay/pass_<k>.pt (distinct states), <run>/replay/index.json
"""
import argparse
import json
from pathlib import Path

import torch

from agent import Kernel
from score import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inner", default="Qwen/Qwen3.5-0.8B")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.load(open(run / "config.json"))
    rows = [json.loads(l) for l in open(run / "log.jsonl")]
    tok, model = load(a.inner)
    k = Kernel(tok, model, cfg["passes"])
    out = run / "replay"
    out.mkdir(exist_ok=True)
    states, used_after = [], []          # (passes_used, edit dict) after each executed cell
    for r in rows:
        res = r.get("result") or ""
        if not res.startswith("OUTPUT of Python run"):
            continue                       # bounced / nudged / final turns ran no code
        code = r["reply"][r["reply"].find("```python\n") + len("```python\n"):]
        code = code[:code.find("```")]
        k.run(code)
        used = cfg["passes"] - k.passes
        states.append((used, {n: k.params[n].detach().cpu().clone() for n in k.orig}))
        print(f"turn {r['turn']}: passes used {used} (logged {cfg['passes'] - r.get('passes_left', 0)}), "
              f"edited {len(k.orig)} tensors", flush=True)
    index, last_saved = {}, None
    for budget in range(0, cfg["passes"] + 1):
        cand = [s for s in states if s[0] <= budget]
        edit = cand[-1][1] if cand else {}
        sig = tuple(sorted((n, round(float(t.float().norm()), 3)) for n, t in edit.items()))
        if sig != last_saved:
            torch.save(edit, out / f"pass_{budget}.pt")
            last_saved, index[budget] = sig, f"pass_{budget}.pt"
        else:
            index[budget] = index[budget - 1]
    real = torch.load(run / "edit.pt", map_location="cpu")
    final = states[-1][1] if states else {}
    same_keys = set(real) == set(final)
    max_rel = max(((real[n].float() - final[n].float()).norm() / real[n].float().norm()).item() for n in real) \
        if same_keys and real else None
    index["_check"] = {"same_tensors_as_real_edit": same_keys, "max_rel_diff_vs_real_edit": max_rel}
    json.dump(index, open(out / "index.json", "w"), indent=1)
    print("index:", index, flush=True)


if __name__ == "__main__":
    main()
