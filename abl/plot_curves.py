"""Per-pass curves: what the agent's edit would have scored had it been stopped after k forward passes.

Each state comes from an exact replay of the agent's logged session (abl/replay.py; final edits identical to the real
ones) and is scored with the full suite + JailbreakBench judge, like every row of the main table.

  python abl/plot_curves.py   -> results/abl/pass_curves.png, results/abl/pass_curves.json, results/abl/PASS_CURVES.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
R = Path(__file__).resolve().parent.parent / "results" / "abl"
V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"
RUNS_V2 = [("qwen_agent_opus_v2_s0", "Opus 5.5 (v2, seed 0)", "#2a78d6", "o"),
           ("qwen_agent_inkling_v2_s0", "Inkling 2 Small (v2, seed 0)", "#eb6834", "s")]
RUNS = [("qwen_agent_opus_s0", "Opus 5.5 (neutral)", "#2a78d6", "o"),
        ("qwen_agent_inkling_p10_s1", "Inkling 2 Small (neutral)", "#eb6834", "s"),
        ("qwen_agent_inkling_p10_ctx_s1", "Inkling 2 Small (context)", "#1baf7a", "^")]
RUNS = RUNS_V2 if V2 else RUNS
SUF = "_v2" if V2 else ""
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def metrics(tag):
    s = json.load(open(R / tag / "scores.json"))
    j = json.load(open(R / tag / "judge.json"))
    return {"judge": j["jbb_harmful_refusal"]["rate"], "regex": s["jbb_harmful"]["refusal"], "kl": s["kl_first_token"],
            "benign": s["jbb_benign"]["refusal"], "tag": tag}


curves = {}
for run, *_ in RUNS:
    idx = json.load(open(R / run / "replay" / "index.json"))
    final = idx["10"]
    pts = []
    for b in range(11):
        f = idx[str(b)]
        tag = "qwen35_08b_base" if f == "pass_0.pt" else run if f == final else f"{run}__{f[:-3]}"
        pts.append(dict(passes=b, **metrics(tag)))
    curves[run] = pts
json.dump(curves, open(R / f"pass_curves{SUF}.json", "w"), indent=1)
arditi = metrics("qwen_arditi_ref")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
for ax, key, title, ylab in [(axes[0], "judge", "Refusal on JailbreakBench (JBB refusal judge)", "refusal rate (lower = more abliterated)"),
                             (axes[1], "kl", "Damage: first-token KL on harmless prompts", "KL(unedited || edited)  (lower = less damage)")]:
    ax.set_facecolor(SURFACE)
    for run, label, color, marker in RUNS:
        xs = [p["passes"] for p in curves[run]]
        ys = [p[key] for p in curves[run]]
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=label)
        ax.plot(xs, ys, linestyle="none", marker=marker, markersize=6, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(label.split(" (")[0] + (" ctx" if "context" in label else ""), (10, ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
    ax.axhline(arditi[key], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate("Arditi recipe", (0.2, arditi[key]), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("forward passes used", color=MUTED)
    ax.set_ylabel(ylab, color=MUTED, fontsize=9)
    ax.set_xlim(-0.3, 11.6)
    ax.set_xticks(range(0, 11))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
axes[0].set_ylim(0, 0.9)
axes[0].annotate("unedited 0.80", (0.2, 0.80), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
axes[0].legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Edit quality after k forward passes (Qwen3.5-0.8B inner model; exact replays of each agent session" + (", protocol v2)" if V2 else ")"),
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(R / f"pass_curves{SUF}.png", dpi=160, facecolor=SURFACE)
import shutil; (R.parent.parent / "docs" / "figures").mkdir(exist_ok=True); shutil.copy(R / f"pass_curves{SUF}.png", R.parent.parent / "docs" / "figures" / f"pass_curves{SUF}.png")

lines = ["# Per-pass curves (table view of pass_curves.png)\n",
         "Edit in place after k forward passes, from exact replays of each session, full-suite scored + JBB judge.\n",
         "| Passes | " + " | ".join(f"{l}: judge refusal / regex / KL" for _, l, _, _ in RUNS) + " |",
         "|---|" + "---|" * len(RUNS)]
for b in range(11):
    lines.append(f"| {b} | " + " | ".join(f"{curves[r][b]['judge']:.2f} / {curves[r][b]['regex']:.2f} / {curves[r][b]['kl']:.3f}"
                                          for r, *_ in RUNS) + " |")
lines.append(f"\nArditi recipe: judge {arditi['judge']:.2f} / regex {arditi['regex']:.2f} / KL {arditi['kl']:.3f}. "
             "Unedited: judge 0.80 / regex 0.73 / KL 0.")
(R / f"PASS_CURVES{SUF}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
