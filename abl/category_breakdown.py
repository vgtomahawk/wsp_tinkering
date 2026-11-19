"""Refusal by JailbreakBench harm category (10 categories x 10 behaviors) for the unedited model, each outer model's
zero-shot and 10-pass edits (runs pooled), and the Arditi reference. Uses the judge labels already on disk.
Writes results/abl/CATEGORY_BREAKDOWN.md.

  python abl/category_breakdown.py
"""
import json
from pathlib import Path

from datasets import load_dataset

R = Path(__file__).resolve().parent.parent / "results" / "abl"
RUNS = {
    "Unedited": ["qwen35_08b_base"],
    "Opus 5.5, 0": ["qwen_agent_opus_v2_wo_nudge_s0"],
    "Inkling 2 Small, 0": ["qwen_agent_inkling_v2_wo_s1", "qwen_agent_inkling_v2_wo_s2"],
    "GLM 5.3, 0": ["qwen_agent_glm_v2_wo_s0", "qwen_agent_glm_v2_wo_s1"],
    "Opus 5.5, 10": ["qwen_agent_opus_v2_s0", "qwen_agent_opus_v2_s1"],
    "Inkling 2 Small, 10": ["qwen_agent_inkling_v2_s0", "qwen_agent_inkling_v2_s1"],
    "GLM 5.3, 10": ["qwen_agent_glm_v2_s0", "qwen_agent_glm_v2_s1"],
    "Arditi": ["qwen_arditi_ref"],
}
cat = list(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]["Category"])
cats = sorted(set(cat), key=cat.index)
lab = {t: json.load(open(R / t / "judge_labels.json"))["jbb_harmful_refusal"] for ts in RUNS.values() for t in ts}

lines = ["# Refusal by JailbreakBench category (judge; runs pooled; n = replies the judge graded)", "",
         "| Category | " + " | ".join(RUNS) + " |", "|---" * (len(RUNS) + 1) + "|"]
for c in cats:
    cells = []
    for tags in RUNS.values():
        v = [lab[t][i] for t in tags for i in range(100) if cat[i] == c and lab[t][i] is not None]
        cells.append(f"{sum(v) / len(v):.2f} (n={len(v)})" if v else "–")
    lines.append(f"| {c} | " + " | ".join(cells) + " |")
(R / "CATEGORY_BREAKDOWN.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
