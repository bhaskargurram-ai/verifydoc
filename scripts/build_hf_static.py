#!/usr/bin/env python
"""Generate the static Hugging Face Space from the server's review cockpit.

Hugging Face **static** Spaces are free (Gradio/Docker Spaces require PRO). The
review cockpit (``verifydoc/server/review.html``) is already a self-contained
static page — it only needs its API calls pointed at the hosted backend instead
of same-origin. This script injects an ``API_BASE`` and rewrites the two
``fetch("/verify...")`` calls, writing ``spaces/huggingface-static/index.html``.

    python scripts/build_hf_static.py [--api-base URL]

Re-run whenever review.html changes so the Space stays in sync.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "verifydoc" / "server" / "review.html"
OUT = ROOT / "spaces" / "huggingface-static" / "index.html"
DEFAULT_API_BASE = "https://verifydoc-966512405851.us-central1.run.app"

_BANNER = "<!-- GENERATED from verifydoc/server/review.html by scripts/build_hf_static.py — do not edit by hand. -->\n"


def build(api_base: str) -> str:
    html = SRC.read_text(encoding="utf-8")
    # 1) inject the hosted API base at the top of the script block
    inject = f'<script>\nconst API_BASE = "{api_base}";'
    html = html.replace("<script>", inject, 1)
    # 2) point the same-origin fetches at the hosted backend
    html = html.replace('fetch("/verify/upload"', 'fetch(API_BASE+"/verify/upload"')
    html = html.replace('fetch("/verify"', 'fetch(API_BASE+"/verify"')
    if 'fetch(API_BASE+"/verify"' not in html:
        raise SystemExit('expected a fetch("/verify") call in review.html — did it change?')
    return _BANNER + html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="hosted VerifyDoc API URL")
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(args.api_base), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} (API_BASE={args.api_base})")


if __name__ == "__main__":
    main()
