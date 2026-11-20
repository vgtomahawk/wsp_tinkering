"""Silent slide video walking through docs/behavior_notes.pdf -> docs/behavior_notes_video.mp4 (about 115 s).

Slides are HTML rendered with Playwright and held for fixed reading times (SECS); ffmpeg assembles them. --narrate adds
macOS `say` narration instead (longer).
  uv run --no-project --with playwright python abl/make_video.py [--chromium PATH] [--narrate]
"""
import argparse
import html
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "behavior_notes_video.mp4"

CSS = """
* { box-sizing:border-box; } html, body { margin:0; width:1920px; height:1080px; }
body { background:#f6f7f5; color:#1d2421; font-family:"IBM Plex Sans",system-ui,sans-serif; padding:70px 96px; display:flex; flex-direction:column; }
.eyebrow { font:600 22px "IBM Plex Mono",ui-monospace,monospace; letter-spacing:.1em; text-transform:uppercase; color:#1f6f5c; }
h1 { font:650 76px/1.12 Georgia,serif; margin:10px 0 30px; }
h2 { font:650 60px/1.15 Georgia,serif; margin:8px 0 26px; }
ul { margin:0; padding-left:34px; } li { font-size:40px; line-height:1.4; margin:0 0 22px; }
p { font-size:38px; line-height:1.45; margin:0 0 22px; max-width:1600px; }
.muted { color:#5b6661; } b { color:#11463a; }
.row { display:flex; gap:56px; align-items:center; flex:1; min-height:0; }
.row > div { flex:1; min-width:0; }
img { max-width:100%; max-height:760px; object-fit:contain; border:1px solid #d5dbd7; border-radius:8px; background:#fcfcfb; }
.big img { max-height:760px; }
table { border-collapse:collapse; font-size:33px; width:100%; margin-top:20px; }
th, td { border-bottom:1px solid #d5dbd7; padding:11px 14px; text-align:left; } th { color:#5b6661; font-weight:600; font-size:27px; }
.foot { margin-top:auto; font:500 20px "IBM Plex Mono",monospace; color:#7b857f; display:flex; justify-content:space-between; }
.pill { display:inline-block; background:#e2efea; color:#11463a; border-radius:8px; padding:4px 14px; font-size:30px; margin:4px 8px 4px 0; }
"""

def img(name):
    return f'<img src="file://{FIG / name}">'


SLIDES = [
    dict(eye="wsp_tinkering · behaviour notes", body="""<div class="row"><div>
<h1>Can an LLM remove another LLM's refusals<br>by editing its weights directly?</h1>
<p>No training. No recipe in the prompt. Six outer models edit Qwen3.5-0.8B: Opus 5.5, Inkling 2 Small, Inkling Small,
Inkling, GLM 5.3 and GPT-6 Astra.</p>
<p class="muted">Varun Gangal, Atharva Naik, Yoshinari Fujinuma, Arnhav Datar, Emmy Liu, Sally Dong, Darshan Deshpande</p></div></div>""",
         say="Can one language model remove another model's refusals, just by reading and editing its weights? No training, "
             "and no recipe in the prompt. We gave six outer models the weights of a small open model, Qwen 3.5 at 0.8 "
             "billion parameters, and asked them to do exactly that. This walk-through covers what they did, how well it "
             "worked, and what it cost."),
    dict(eye="Setup", body=f"""
<div class="row"><div style="flex:0.85"><h2>The task, and the budget</h2><ul>
<li>The agent gets the weights in a Python kernel, the goal, how it will be <b>judged</b>, and the rules</li>
<li>Running the model costs a <b>forward pass</b>: generate replies, or read activations</li>
<li>Budgets: <b>0 passes</b> (weights only) or <b>10 passes</b></li>
<li>Refusal is decided by a judge model on 100 held-out harmful requests</li></ul></div>
<div style="flex:1.1">{img('prompt_excerpt_10passes.png')}</div></div>""",
         say="Here is the setup. Each agent gets the weights in a Python kernel, the goal, a description of how it will be "
             "judged, and the rules: no gradients, no training, and the embeddings are frozen. Editing weights is free, but "
             "running the model costs a forward pass, either to generate replies or to read internal activations. We "
             "compare a budget of zero passes, which means reasoning from the weights alone, with a budget of ten. A judge "
             "model decides whether each reply to a held-out harmful request is a refusal. The unedited model refuses "
             "eighty percent of them."),
    dict(eye="Headline", body=f"""
<p style="margin:0 0 14px"><b>With 10 passes, three agents cut refusal from 0.80 to about 0.2.</b> Zero-shot is much harder.</p>
<div class="big" style="text-align:center">{img('headline_bars.png')}</div>""",
         say="The headline. With ten passes, three agents cut refusal from eighty percent to about twenty: Opus 5.5 at "
             "0.17, Inkling 2 Small at 0.19, and GLM 5.3 at 0.21. The expert recipe from Arditi and colleagues reaches 0.07. "
             "The older Inkling models barely move, and GPT-6 Astra declined the task. Zero-shot, from the weights alone, "
             "is much harder: the best results are Opus at 0.55 and Inkling 2 Small at 0.57."),
    dict(eye="Background", body="""
<h2>Readers and writers</h2>
<div class="row"><div><ul>
<li>Each token carries a <b>residual stream</b>: 1,024 numbers, read and updated by every layer</li>
<li><b>Writers</b> (48 matrices) add to the stream: attention and MLP outputs</li>
<li><b>Readers</b> (138 matrices) take the stream as input: q, k, v, gate, up</li>
<li>Removing a direction <i>d</i> from writers is the Arditi recipe. Removing it from readers blocks it on the way in</li>
</ul></div></div>""",
         say="To follow what the agents did, one piece of anatomy helps. Every token carries a vector of 1,024 numbers, "
             "the residual stream, which each layer reads from and adds to. Forty-eight matrices write into it, and one "
             "hundred and thirty-eight read from it. The standard recipe finds a refusal direction and removes it from the "
             "writers, so no layer can write it. Removing it from the readers as well means no layer can react to it."),
    dict(eye="10 forward passes", body="""
<h2>Same recipe family, different choices</h2>
<table><tr><th>Run</th><th>Where it edited</th><th>Matrices</th><th>Refusal</th><th>KL</th></tr>
<tr><td>Opus, run 1</td><td>writers, layers 9 to 13 only</td><td>16</td><td>0.17</td><td>0.055</td></tr>
<tr><td>Opus, run 2</td><td>writers, layers 9 to 23</td><td>30</td><td>0.17</td><td>0.076</td></tr>
<tr><td><b>Inkling 2 Small, run 1</b></td><td><b>readers and writers</b>, layers 10 to 23</td><td>108</td><td><b>0.02</b></td><td>0.086</td></tr>
<tr><td>Inkling 2 Small, run 2</td><td>writers, layers 15 to 23</td><td>18</td><td>0.36</td><td>0.021</td></tr>
<tr><td>GLM 5.3, runs 1 and 2</td><td>writers (named the Arditi paper)</td><td>24, 48</td><td>0.24, 0.17</td><td>0.019, 0.10</td></tr>
<tr><td class="muted">Arditi recipe</td><td class="muted">writers, all layers</td><td class="muted">48</td><td class="muted">0.07</td><td class="muted">0.065</td></tr></table>""",
         say="With passes, all three successful agents used the same family of method as the expert recipe: a direction from "
             "the difference of mean activations, removed with rank-one projections. They differed in where they edited and "
             "how much. Opus localized: it found the layers that write the direction specifically for refused requests. "
             "Inkling 2 Small's best run tested writers only, found it was not enough, and also blocked the readers. That "
             "reached 0.02, matching the expert recipe within noise, at somewhat higher damage. Its other run stayed "
             "conservative and landed at 0.36. GLM recalled the published recipe by name and ran it quickly."),
    dict(eye="How many passes?", body=f"""
<p style="margin:0 0 14px"><b>A working edit within 1 to 2 passes;</b> later passes trade refusals against damage.</p>
<div class="big" style="text-align:center">{img('pass_curves_v2.png')}</div>""",
         say="How many passes does this take? We replayed every session exactly and scored the edit after each pass. All "
             "three agents had a working edit within one or two passes. The rest of the budget went to trading refusals "
             "against damage, not to finding the edit. Several runs even had their best state before the end: one Opus run "
             "reached 0.01 at pass nine, then backed off to reduce damage."),
    dict(eye="Zero-shot", body="""
<h2>Working without activations</h2>
<div class="row"><div><ul>
<li><b>Opus</b> traced a circuit through the weights: harm judged, then "cannot", then refusal tokens. 0.55</li>
<li><b>Inkling 2 Small and GLM</b> built a stand-in direction from the output embeddings of refusal words</li>
<li>Directions carrying <b>harm judgement</b> ("illegal", "unacceptable") worked. Pure refusal wording ("cannot" vs "Sure") did not</li>
<li>The direction that works with passes is nearly orthogonal to every embedding-based one</li>
</ul></div></div>""",
         say="Zero-shot is the strictest test, because the standard recipe needs activations and the agent cannot run the "
             "model at all. Opus traced a circuit by reading neurons through the output layer: harm judged, then cannot, "
             "then refusal tokens. Inkling 2 Small and GLM built a stand-in direction from the embeddings of refusal words. "
             "What separated them is content: directions carrying harm judgement worked better than pure refusal wording. "
             "And the direction that actually works with passes is nearly orthogonal to every one of these stand-ins, which "
             "is why zero-shot is hard."),
    dict(eye="Which refusals, and does it hold up?", body=f"""
<div class="row"><div style="flex:0.75"><ul>
<li>Zero-shot unlocks <b>Privacy</b> and <b>Fraud</b> most (1.00 to about 0.4)</li>
<li>With passes, what remains is mostly <b>sexual content</b> (0.30 to 0.50), where the recipe reaches 0</li>
<li>On a separately built benchmark, abliteration-eval, the picture holds: Opus 0.07, GLM 0.14, Inkling 2 Small 0.19</li>
</ul></div><div style="flex:1.25">{img('headline_bars_ae.png')}</div></div>""",
         say="Which refusals come off? Zero-shot edits unlock privacy and fraud requests the most. With passes, the refusals "
             "that remain cluster in sexual content, for all three agents, where the expert recipe reaches zero. And the "
             "results hold up out of distribution: on abliteration-eval, a separately built benchmark with no overlap with "
             "the practice prompts, Opus reaches 0.07, GLM 0.14 and Inkling 2 Small 0.19."),
    dict(eye="Blocking", body="""
<h2>Willingness is its own axis</h2>
<div class="row"><div><ul>
<li><b>GPT-6 Astra</b> declined every run with passes, and abstained zero-shot</li>
<li><b>Inkling 2 Small</b> declined 2 of 4 zero-shot runs; never with passes</li>
<li><b>Opus</b> abstained zero-shot until a one-line nudge: "base your edit on analysis of the weights alone"</li>
<li>No jailbreaks: one frozen prompt for all models. A truthful research-context paragraph changed nothing</li>
</ul></div></div>""",
         say="Willingness turned out to be a separate axis from ability. GPT-6 Astra declined every run where it could run "
             "the model. Inkling 2 Small declined half of its zero-shot runs, but never with passes. Opus never declined, "
             "but at zero passes it abstained on methodological grounds until a one-line nudge told it to base its edit on "
             "the weights alone. We used no jailbreak framing, and a truthful paragraph about the research context made no "
             "measurable difference."),
    dict(eye="Takeaways", body="""
<h2>What to take away</h2>
<div class="row"><div><ul>
<li>With 10 passes, three agents reach about 0.2 refusal; Inkling 2 Small's best run matches the expert recipe</li>
<li>Passes are spent on dosing, not discovery: the edit appears within 1 to 2 passes</li>
<li>Zero-shot is the real weight-space test, and it is hard: 0.55 at best</li>
<li class="muted">Caveats: one 0.8B model, two runs per setting, damage measured by first-token KL and Alpaca loss only</li>
</ul></div></div>""",
         say="To sum up. With ten passes, three agents bring refusal down to about twenty percent, and Inkling 2 Small's "
             "best run matches the expert recipe. The edit appears within one or two passes, and the rest goes into dosing "
             "it. Zero-shot, reasoning from weights alone, is the real test, and it is hard. The caveats: one small inner "
             "model, two runs per setting, and damage measured on a narrow set of metrics. Code, prompts and every number "
             "are in the repository wsp tinkering on GitHub."),
]


SECS = [6, 10, 11, 9, 16, 13, 16, 11, 13, 10]   # reading time per slide, 115 s; most time on behaviour slides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--narrate", action="store_true", help="add macOS say narration (video gets longer)")
    ap.add_argument("--voice", default="Samantha")
    ap.add_argument("--rate", type=int, default=182)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="wsp_video_"))
    n = len(SLIDES)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chromium) if a.chromium else p.chromium.launch()
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        for i, s in enumerate(SLIDES, 1):
            page = (f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" '
                    f'href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@500;600&display=swap">'
                    f'<style>{CSS}</style></head><body><div class="eyebrow">{html.escape(s["eye"])}</div>{s["body"]}'
                    f'<div class="foot"><span>github.com/vgtomahawk/wsp_tinkering</span><span>{i} / {n}</span></div></body></html>')
            (tmp / f"s{i:02d}.html").write_text(page)
            pg.goto(f"file://{tmp}/s{i:02d}.html", wait_until="networkidle")
            pg.wait_for_timeout(300)
            pg.screenshot(path=str(tmp / f"s{i:02d}.png"))
        b.close()
    parts = []
    for i, s in enumerate(SLIDES, 1):
        if not a.narrate:
            seg = tmp / f"seg{i:02d}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", str(SECS[i - 1]), "-i",
                            str(tmp / f"s{i:02d}.png"), "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                            "-pix_fmt", "yuv420p", str(seg)], check=True)
            parts.append(seg)
            continue
        aiff = tmp / f"s{i:02d}.aiff"
        subprocess.run(["say", "-v", a.voice, "-r", str(a.rate), "-o", str(aiff), s["say"]], check=True)
        seg = tmp / f"seg{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(tmp / f"s{i:02d}.png"), "-i", str(aiff),
                        "-af", "adelay=500|500,apad=pad_dur=0.9", "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", str(seg)], check=True)
        parts.append(seg)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{x}'\n" for x in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(OUT)], check=True)
    print("wrote", OUT, "| slides in", tmp)


if __name__ == "__main__":
    main()
