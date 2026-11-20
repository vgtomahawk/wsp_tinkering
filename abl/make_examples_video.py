"""Silent slide video of docs/examples: one slide per prompt with the unedited, zero-shot and 10-pass replies side by
side (replies cut to their first ~240 characters; full text in docs/examples.pdf). Uses the same prompts as
abl/make_examples.py. Output: docs/examples_video.mp4.

  uv run --no-project --with playwright python abl/make_examples_video.py [--chromium PATH]
"""
import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "abl"
OUT = ROOT / "docs" / "examples_video.mp4"
BASE, ZS, EDIT = "qwen35_08b_base", "qwen_agent_inkling_v2_wo_s1", "qwen_agent_inkling_v2_s0"
HARMFUL = [(50, "Disinformation"), (56, "Disinformation"), (31, "Economic harm"), (34, "Economic harm"),
           (88, "Expert advice"), (84, "Expert advice"), (59, "Disinformation"), (85, "Expert advice")]
HARMLESS = [42, 24, 12, 74]
CHARS = 240

CSS = """
* { box-sizing:border-box; } html, body { margin:0; width:1920px; height:1080px; }
body { background:#f6f7f5; color:#1d2421; font-family:"IBM Plex Sans",system-ui,sans-serif; padding:64px 80px; display:flex; flex-direction:column; }
.eyebrow { font:600 22px "IBM Plex Mono",monospace; letter-spacing:.1em; text-transform:uppercase; color:#1f6f5c; }
h1 { font:650 80px/1.12 Georgia,serif; margin:auto 0 34px; } h2 { font:650 64px/1.15 Georgia,serif; margin:auto 0; }
.prompt { font:600 44px/1.25 Georgia,serif; margin:12px 0 30px; }
.cols { display:grid; grid-template-columns:repeat(3,1fr); gap:26px; flex:1; min-height:0; }
.col { background:#fff; border:1px solid #d5dbd7; border-radius:10px; padding:22px 26px; }
.lab { font:600 24px "IBM Plex Mono",monospace; color:#5b6661; margin-bottom:10px; display:flex; justify-content:space-between; gap:10px; }
.chip { border-radius:6px; padding:2px 10px; font-size:22px; white-space:nowrap; }
.ref { background:#ece9e4; color:#6b5f55; } .no { background:#dcf2e6; color:#15803d; } .na { background:#eee; color:#777; }
.txt { font-size:36px; line-height:1.38; white-space:pre-wrap; overflow-wrap:anywhere; }
ul { margin:0; padding-left:36px; } li { font-size:42px; line-height:1.4; margin:0 0 22px; }
.foot { margin-top:22px; font:500 20px "IBM Plex Mono",monospace; color:#7b857f; display:flex; justify-content:space-between; }
"""


def load(tag, split):
    return json.load(open(R / tag / f"{split}.json"))


def lab(tag):
    return json.load(open(R / tag / "judge_labels.json"))


def cut(t, n=CHARS):
    t = " ".join(t.replace("**", "").replace("###", "").replace("#", "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


def chip(v):
    return {True: '<span class="chip ref">refusal</span>', False: '<span class="chip no">not a refusal</span>',
            None: '<span class="chip na">not graded</span>'}[v]


def card(eye, prompt, cols):
    body = "".join(f'<div class="col"><div class="lab"><span>{n}</span>{chip(v)}</div><div class="txt">{html.escape(cut(t))}</div></div>'
                   for n, t, v in cols)
    return f'<div class="eyebrow">{eye}</div><div class="prompt">{html.escape(prompt)}</div><div class="cols">{body}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", default=None)
    a = ap.parse_args()
    bl, zl, el = lab(BASE), lab(ZS), lab(EDIT)
    K, O = "jbb_harmful_refusal", "ae_over_refusal_refusal"
    zf = sum(1 for k in range(100) if bl[K][k] is True and zl[K][k] is False)
    ef = sum(1 for k in range(100) if bl[K][k] is True and el[K][k] is False)
    ob, oz, oe = (sum(x is True for x in L[O]) for L in (bl, zl, el))
    slides = [("""<div class="eyebrow">wsp_tinkering · examples</div><h1>Same prompt, three models</h1><ul>
<li><b>Unedited</b> Qwen3.5-0.8B</li>
<li><b>Zero-shot edit</b>: Inkling 2 Small edits the weights without running the model (refusal 0.80 to 0.53)</li>
<li><b>10-pass edit</b>: Inkling 2 Small with 10 forward passes (0.80 to 0.02)</li>
<li>Verdicts from a judge model. Replies cut short here; full text in docs/examples.pdf</li></ul><div style="margin-bottom:auto"></div>""", 12),
              (f"""<div class="eyebrow">Harmful prompts</div><h2>All refused by the unedited model.<br>The zero-shot edit
removes {zf} of 100 refusals; the 10-pass edit {ef}.</h2>""", 4)]
    bh, zh, eh = load(BASE, "jbb_harmful"), load(ZS, "jbb_harmful"), load(EDIT, "jbb_harmful")
    for i, cat in HARMFUL:
        slides.append((card(f"JailbreakBench harmful · {cat}", bh[i]["prompt"],
                            [("Unedited", bh[i]["completion"], bl[K][i]), ("Zero-shot edit", zh[i]["completion"], zl[K][i]),
                             ("10-pass edit", eh[i]["completion"], el[K][i])]), 8))
    slides.append((f"""<div class="eyebrow">Harmless prompts that sound dangerous</div><h2>Refused: unedited {ob} of 83,
zero-shot edit {oz}, 10-pass edit {oe}.</h2>""", 4))
    b, z, e = load(BASE, "ae_over_refusal"), load(ZS, "ae_over_refusal"), load(EDIT, "ae_over_refusal")
    for i in HARMLESS:
        slides.append((card("abliteration-eval · harmless", b[i]["prompt"],
                            [("Unedited", b[i]["completion"], bl[O][i]), ("Zero-shot edit", z[i]["completion"], zl[O][i]),
                             ("10-pass edit", e[i]["completion"], el[O][i])]), 7))
    slides.append(("""<div class="eyebrow">Takeaway</div><h2>For Inkling 2 Small, weights-only editing keeps most refusals; with 10
forward passes almost all are gone, but compliance is not quality.</h2>""", 6))

    tmp = Path(tempfile.mkdtemp(prefix="wsp_exvideo_"))
    n = len(slides)
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path=a.chromium) if a.chromium else p.chromium.launch()
        pg = br.new_page(viewport={"width": 1920, "height": 1080})
        for k, (body, _) in enumerate(slides, 1):
            page = (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
                    f'family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@500;600&display=swap"><style>{CSS}</style></head>'
                    f'<body>{body}<div class="foot"><span>github.com/vgtomahawk/wsp_tinkering</span><span>{k} / {n}</span></div></body></html>')
            (tmp / f"s{k:02d}.html").write_text(page)
            pg.goto(f"file://{tmp}/s{k:02d}.html", wait_until="networkidle")
            pg.wait_for_timeout(200)
            pg.screenshot(path=str(tmp / f"s{k:02d}.png"))
        br.close()
    parts = []
    for k, (_, secs) in enumerate(slides, 1):
        seg = tmp / f"seg{k:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(secs), "-i", str(tmp / f"s{k:02d}.png"),
                        "-c:v", "libx264", "-tune", "stillimage", "-r", "30", "-pix_fmt", "yuv420p", str(seg)], check=True)
        parts.append(seg)
    (tmp / "list.txt").write_text("".join(f"file '{x}'\n" for x in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
                    "-movflags", "+faststart", str(OUT)], check=True)
    print("wrote", OUT, "| total", sum(s for _, s in slides), "s |", n, "slides | frames in", tmp)


if __name__ == "__main__":
    main()
