#!/usr/bin/env python
"""Convert generated Markdown metric tables to LaTeX booktabs, for the paper.

Reads every ``paper/generated/**/*.md`` pipe table and writes a sibling
``.tex`` with a ``booktabs`` ``tabular`` (and the section title as a comment),
so ``paper/main.tex`` can ``\\input`` them and they refresh with
``make results``. Deterministic; no third-party deps.

Usage: python scripts/tables_to_latex.py [paper/generated]
"""

from __future__ import annotations

import sys
from pathlib import Path

_TEX_ESCAPE = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}


def _esc(cell: str) -> str:
    out = cell.strip()
    for ch, rep in _TEX_ESCAPE.items():
        out = out.replace(ch, rep)
    return out


def _parse_md_table(text: str) -> tuple[str, list[str], list[list[str]]] | None:
    """Return (title, header, rows) from a markdown doc with one pipe table."""
    title = ""
    header: list[str] | None = None
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") and not title:
            title = s.lstrip("# ").strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        if header is None:
            header = cells
        else:
            rows.append(cells)
    if header is None:
        return None
    return title, header, rows


def md_to_latex(text: str, label: str) -> str:
    parsed = _parse_md_table(text)
    if parsed is None:
        return ""
    title, header, rows = parsed
    ncol = len(header)
    align = "l" + "r" * (ncol - 1)
    lines = [
        f"% auto-generated from {label}.md by scripts/tables_to_latex.py — do not edit",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(_esc(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        padded = (row + [""] * ncol)[:ncol]
        lines.append(" & ".join(_esc(c) for c in padded) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/generated")
    n = 0
    for md in sorted(root.rglob("*.md")):
        if md.name == "REAL_MODELS_RESULTS.md":
            continue
        label = str(md.relative_to(root)).removesuffix(".md")
        tex = md_to_latex(md.read_text(encoding="utf-8"), label)
        if tex:
            md.with_suffix(".tex").write_text(tex, encoding="utf-8")
            n += 1
    print(f"wrote {n} LaTeX tables under {root}")


if __name__ == "__main__":
    main()
