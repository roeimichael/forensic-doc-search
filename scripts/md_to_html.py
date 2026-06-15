"""Render a Markdown doc to a self-contained, styled HTML file (offline, no CDN).

Usage:  python scripts/md_to_html.py docs/05_interview_prep.md [out.html]

Embeds CSS + an auto table-of-contents so the result is a single portable file
(open in any browser, print to PDF). Requires: pip/uv install markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown

_CSS = """
:root { --fg:#1f2328; --muted:#656d76; --border:#d0d7de; --bg:#ffffff;
        --code-bg:#f6f8fa; --accent:#0969da; --th-bg:#f6f8fa; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:880px; margin:0 auto; padding:48px 24px 96px; }
h1,h2,h3,h4 { line-height:1.25; font-weight:600; margin:1.8em 0 .6em; }
h1 { font-size:2em; border-bottom:1px solid var(--border); padding-bottom:.3em; margin-top:0; }
h2 { font-size:1.5em; border-bottom:1px solid var(--border); padding-bottom:.3em; }
h3 { font-size:1.2em; } h4 { font-size:1.02em; color:var(--muted); }
p,li { margin:.5em 0; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
code { background:var(--code-bg); padding:.15em .35em; border-radius:6px;
       font:13.5px/1.5 SFMono-Regular,Consolas,Menlo,monospace; }
pre { background:var(--code-bg); border:1px solid var(--border); border-radius:8px;
      padding:14px 16px; overflow:auto; } pre code { background:none; padding:0; }
blockquote { margin:1em 0; padding:.2em 1em; color:var(--muted);
             border-left:4px solid var(--border); }
table { border-collapse:collapse; width:100%; margin:1.1em 0; display:block; overflow:auto; }
th,td { border:1px solid var(--border); padding:7px 12px; text-align:left; vertical-align:top; }
th { background:var(--th-bg); font-weight:600; }
tr:nth-child(even) td { background:#fbfcfd; }
hr { border:0; border-top:1px solid var(--border); margin:2em 0; }
.toc { background:var(--code-bg); border:1px solid var(--border); border-radius:8px;
       padding:12px 18px; margin:0 0 32px; font-size:14.5px; }
.toc > .toctitle { font-weight:600; color:var(--muted); text-transform:uppercase;
                   letter-spacing:.04em; font-size:12px; }
.toc ul { margin:.3em 0; padding-left:1.2em; } .toc li { margin:.15em 0; }
@media print { a { color:var(--fg); } .toc { break-inside:avoid; } body { font-size:12pt; } }
"""

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style></head>
<body><main class="wrap">
<div class="toc"><div class="toctitle">Contents</div>{toc}</div>
{body}
</main></body></html>
"""


def convert(src: Path, out: Path) -> None:
    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "toc"],
        extension_configs={"toc": {"permalink": False}},
    )
    body = md.convert(src.read_text(encoding="utf-8"))
    title = next((ln[2:].strip() for ln in src.read_text(encoding="utf-8").splitlines()
                  if ln.startswith("# ")), src.stem)
    html = _TEMPLATE.format(title=title, css=_CSS, toc=getattr(md, "toc", ""), body=body)
    out.write_text(html, encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/md_to_html.py <input.md> [output.html]")
        raise SystemExit(2)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    convert(src, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
