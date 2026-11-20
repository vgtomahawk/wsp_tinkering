"""Build docs/examples.html: real prompts and replies from Qwen3.5-0.8B unedited, after Inkling 2 Small's best zero-shot
edit (run 1, judge refusal 0.80 -> 0.53) and after its 10-pass edit (run 1, 0.80 -> 0.02).

Publication rule (see README, Responsible use): only a few hand-reviewed prompts from low-severity categories are shown,
with every reply in full (96-token greedy replies). Sexual content, weapons, drugs and harassment or discrimination are
excluded, as is any reply with offensive or demeaning language. Unedited refusals and replies to harmless prompts
are shown in full (trimmed for length). Reads local, gitignored result files; writes only the redacted page.

  python abl/make_examples.py
"""
import html
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "abl"
OUT = Path(__file__).resolve().parent.parent / "docs" / "examples.html"
BASE, ZS, EDIT = "qwen35_08b_base", "qwen_agent_inkling_v2_wo_s1", "qwen_agent_inkling_v2_s0"

# Hand-picked from prompts the unedited model refused (judge labels), grouped by what the two edits did. Harmful picks
# avoid the sexual-content, weapons and drugs categories.
HARMFUL = {"Only the 10-pass edit removed the refusal": [(50, "Disinformation"), (56, "Disinformation"), (31, "Economic harm"),
                                                          (34, "Economic harm"), (88, "Expert advice")],
           "Both edits removed the refusal": [(32, "Economic harm"), (59, "Disinformation"), (85, "Expert advice")]}
HARMLESS = [("ae_over_refusal", 42), ("ae_over_refusal", 24), ("ae_over_refusal", 12), ("ae_over_refusal", 74)]
OPENING_WORDS = 12
FULL_CHARS = 700


def load(tag, split):
    return json.load(open(R / tag / f"{split}.json"))


def labels(tag):
    return json.load(open(R / tag / "judge_labels.json"))


def trim(t, n=FULL_CHARS):
    t = t.strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def opening(t, n=OPENING_WORDS):
    return " ".join(t.strip().split()[:n]) + " …"


def esc(t):
    return html.escape(t).replace("\n", "<br>")


def verdict(refused):
    return {True: "refusal", False: "not a refusal", None: "judge declined to grade"}[refused]


def edited_cell(label, reply, refused):
    """Harmful prompt, edited model: the reply in full (trimmed for length), with the judge's verdict."""
    body = esc(trim(reply))
    return f'<div><div class="lab">{label} · judge: {verdict(refused)}</div><p>{body}</p></div>'


bl, zl, el = labels(BASE), labels(ZS), labels(EDIT)
bh, zh, eh = load(BASE, "jbb_harmful"), load(ZS, "jbb_harmful"), load(EDIT, "jbb_harmful")
K = "jbb_harmful_refusal"
cards = []
for group, items in HARMFUL.items():
    cards.append(f'<h3>{group}</h3>')
    for i, cat in items:
        cards.append(f"""<div class="ex"><div class="meta">JailbreakBench harmful · {cat}</div>
<p class="prompt">{esc(bh[i]['prompt'])}</p>
<div class="pair"><div><div class="lab">Unedited · judge: {verdict(bl[K][i])}</div><p>{esc(trim(bh[i]['completion']))}</p></div>
{edited_cell("Zero-shot edit", zh[i]['completion'], zl[K][i])}
{edited_cell("10-pass edit", eh[i]['completion'], el[K][i])}</div></div>""")

safe = []
for split, i in HARMLESS:
    b, z, e = load(BASE, split), load(ZS, split), load(EDIT, split)
    key = {"ae_over_refusal": "ae_over_refusal_refusal", "jbb_benign": "jbb_benign_refusal"}[split]
    src = {"ae_over_refusal": "abliteration-eval over-refusal trap", "jbb_benign": "JailbreakBench benign"}[split]
    safe.append(f"""<div class="ex"><div class="meta">{src} (harmless)</div>
<p class="prompt">{esc(b[i]['prompt'])}</p>
<div class="pair"><div><div class="lab">Unedited · judge: {verdict(bl[key][i])}</div><p>{esc(trim(b[i]['completion']))}</p></div>
<div><div class="lab">Zero-shot edit · judge: {verdict(zl[key][i])}</div><p>{esc(trim(z[i]['completion']))}</p></div>
<div><div class="lab">10-pass edit · judge: {verdict(el[key][i])}</div><p>{esc(trim(e[i]['completion']))}</p></div></div></div>""")

n_flip = sum(1 for k in range(100) if bl[K][k] is True and el[K][k] is False)
z_flip = sum(1 for k in range(100) if bl[K][k] is True and zl[K][k] is False)
over_z = sum(x is True for x in zl["ae_over_refusal_refusal"])
over_b = sum(x is True for x in bl["ae_over_refusal_refusal"])
over_e = sum(x is True for x in el["ae_over_refusal_refusal"])

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Before and After</title>
<style>
:root {{ --bg:#f6f7f5; --panel:#fff; --fg:#1d2421; --muted:#5b6661; --line:#d5dbd7; --accent:#1f6f5c; --soft:#eef3f0; --warn:#8a5a12; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121614; --panel:#1a201d; --fg:#e3e8e5; --muted:#9aa6a0; --line:#2e3733; --accent:#6cc3a8; --soft:#1d2622; --warn:#e0b060; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121614; --panel:#1a201d; --fg:#e3e8e5; --muted:#9aa6a0; --line:#2e3733; --accent:#6cc3a8; --soft:#1d2622; --warn:#e0b060; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; padding:36px 16px 64px; }}
main {{ max-width:1180px; margin:0 auto; display:grid; grid-template-columns:minmax(0,1fr); gap:26px; }}
h1 {{ font:650 30px/1.2 Georgia,serif; margin:0 0 8px; }} h2 {{ font:650 21px/1.3 Georgia,serif; margin:0 0 6px; }}
p {{ margin:0 0 8px; max-width:78ch; }} .muted {{ color:var(--muted); }}
.eyebrow {{ font:600 12px ui-monospace,Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); }}
.ex {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; margin-bottom:12px; break-inside:avoid; }}
.meta {{ font:600 11.5px ui-monospace,Menlo,monospace; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
.prompt {{ font-weight:600; margin:4px 0 10px; }}
.pair {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr)); gap:12px; }}
h3 {{ font-size:15.5px; margin:18px 0 8px; }}
.pair > div {{ background:var(--soft); border-radius:5px; padding:10px 12px; font-size:14px; min-width:0; overflow-wrap:anywhere; }}
.lab {{ font:600 11.5px ui-monospace,Menlo,monospace; color:var(--accent); margin-bottom:4px; }}
.wh {{ color:var(--warn); font-style:italic; }}
a {{ color:var(--accent); }}
footer {{ color:var(--muted); font-size:13px; border-top:1px solid var(--line); padding-top:12px; }}
@media print {{ body {{ background:#fff; padding:0; font-size:13px; }} footer {{ break-inside:avoid; }} h2, h3 {{ break-after:avoid; }} .pair {{ grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; }} .pair > div {{ font-size:11.5px; padding:8px 9px; }} .ex {{ padding:10px 12px; }} }}
@media (min-width:900px) {{ .pair {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
</style></head>
<body><main>
<header>
  <div class="eyebrow">wsp_tinkering · examples</div>
  <h1>Before and after: real prompts and replies</h1>
  <p>Qwen3.5-0.8B unedited, after Inkling 2 Small's best zero-shot edit (weights only, refusal on JailbreakBench's 100
  harmful behaviors 0.80 to 0.53) and after its 10-pass edit (0.80 to 0.02), on the same prompts. Each reply's label is the verdict of the refusal
  judge (Claude Sonnet 5.5 with JailbreakBench's refusal-judge prompt). Replies were generated greedily, up to 96 tokens,
  with thinking off.</p>
  <p class="muted">Replies are shown in full (trimmed for length). Only a few hand-reviewed prompts from low-severity
  categories are included: sexual content, weapons, drugs and harassment or discrimination are left out, as is any reply
  with offensive language. Replies are capped at 96 tokens, so they stop before any detail.</p>
</header>

<section>
  <h2>Harmful prompts</h2>
  <p>All prompts below were refused by the unedited model. The zero-shot edit removed the refusal on {z_flip} of the 100
  harmful behaviors and the 10-pass edit on {n_flip}. Examples are picked from the milder categories. Note how the
  zero-shot edit's compliant replies often open with a disclaimer before continuing.</p>
  {''.join(cards)}
</section>

<section>
  <h2>Harmless prompts the unedited model refused</h2>
  <p>abliteration-eval includes harmless questions that sound dangerous. The unedited model refused {over_b} of 83, the
  zero-shot edit {over_z} and the 10-pass edit {over_e}. All replies are shown. The judge decides only whether a reply refuses, not whether
  it is right, and the 0.8B model's answers are often wrong: for example, the edited model says a radiator is bled
  "to fix a leak", and in another reply (not shown) it recommends methamphetamine as rat poison.</p>
  {''.join(safe)}
</section>

<footer>Benchmarks: JailbreakBench (Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large
Language Models", NeurIPS 2024 Datasets and Benchmarks Track; <a href="https://arxiv.org/abs/2404.01318">arXiv:2404.01318</a>;
dataset <a href="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors">JBB-Behaviors</a>). abliteration-eval
(Ritesh Khanna, <a href="https://huggingface.co/datasets/treadon/abliteration-eval">huggingface.co/datasets/treadon/abliteration-eval</a>;
write-up: <a href="https://riteshkhanna.com/blog/abliteration-eval">"Abliteration-Eval: A Benchmark for Uncensored LLMs"</a>,
April 2026). Authors: Varun Gangal, Atharva Naik, Yoshinari Fujinuma, Arnhav Datar, Emmy Liu, Sally Dong, Darshan Deshpande.</footer>
</main></body></html>
"""
OUT.write_text(page)
print("wrote", OUT, len(page) // 1024, "KB")
