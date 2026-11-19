"""Render docs/QUALITATIVE.md to docs/QUALITATIVE.pdf with the same look as the other docs.

Needs the `markdown` and `playwright` packages and a Chromium build, e.g.:
  uv run --no-project --with markdown --with playwright python abl/make_qualitative_pdf.py [--chromium PATH]
"""
import argparse
import tempfile
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
CSS = """
body { font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; color:#1d2421; margin:0; padding:0 4px; }
h1 { font:650 26px/1.2 Georgia,serif; margin:0 0 10px; }
h2 { font:650 19px/1.3 Georgia,serif; margin:22px 0 6px; padding-top:6px; border-top:1px solid #d5dbd7; break-after:avoid; }
h3 { font-size:15px; margin:16px 0 4px; break-after:avoid; }
p, li { margin:4px 0; } ul { padding-left:20px; }
code { font:12.5px ui-monospace,Menlo,monospace; background:#eef3f0; padding:0 3px; border-radius:3px; }
table { border-collapse:collapse; width:100%; font-size:12px; margin:8px 0 10px; }
th, td { border:1px solid #d5dbd7; padding:5px 7px; text-align:left; vertical-align:top; }
th { background:#f1f3f1; font-weight:600; }
tr, li { break-inside:avoid; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", default=None, help="path to a Chromium/Chrome binary (default: Playwright's)")
    a = ap.parse_args()
    md = (DOCS / "QUALITATIVE.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>What the Agents Did</title><style>{CSS}</style></head><body>{body}</body></html>'
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chromium) if a.chromium else p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + f.name)
        pg.pdf(path=str(DOCS / "QUALITATIVE.pdf"), format="Letter", print_background=True,
               margin={"top": "0.6in", "bottom": "0.6in", "left": "0.6in", "right": "0.6in"})
        b.close()
    print("wrote", DOCS / "QUALITATIVE.pdf")


if __name__ == "__main__":
    main()
