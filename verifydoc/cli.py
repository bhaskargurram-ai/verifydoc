"""CLI: ``verifydoc extract doc.pdf --schema schema.json --target... ``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

import verifydoc
from verifydoc.adapters import get_adapter
from verifydoc.pipeline import DEFAULT_THRESHOLD, verify

app = typer.Typer(name="verifydoc", help="Trust layer for document -> JSON extraction.")


@app.command()
def extract(
    source: Path = typer.Argument(..., help="Document to extract from (pdf/image/txt)."),
    schema: Path = typer.Option(..., "--schema", "-s", help="JSON Schema with x-scoring rules."),
    adapter: str = typer.Option("text-search", "--adapter", "-a", help="Extractor adapter."),
    k: int = typer.Option(1, "--k", help="Self-consistency samples (k > 1 enables consensus)."),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD, "--threshold", "-t", help="Accept-decision confidence cutoff."
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write JSON here (else stdout)."),
) -> None:
    """Extract fields with confidence + grounding + accept/review decisions."""
    result = verify(source, schema, adapter=get_adapter(adapter), k=k, threshold=threshold)
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if out is not None:
        out.write_text(payload + "\n", encoding="utf-8")
        typer.echo(
            f"{result.doc_id}: {result.n_accepted} accepted, "
            f"{result.n_review} to review -> {out}"
        )
    else:
        typer.echo(payload)


@app.command()
def version() -> None:
    """Print the installed VerifyDoc version."""
    typer.echo(verifydoc.__version__)


@app.command()
def iaa(
    label_files: list[Path] = typer.Argument(..., help="Annotator label JSON files (>=2)."),
) -> None:
    """Inter-annotator agreement (Cohen's / Fleiss' kappa) from label files."""
    from verifydoc.labeling import iaa_report, load_annotations

    report = iaa_report(load_annotations(list(label_files)))
    typer.echo(report.interpret())
    for (a, b), k in sorted(report.pairwise_cohen.items()):
        typer.echo(f"  {a} vs {b}: Cohen's kappa = {k:.3f}")


if __name__ == "__main__":  # pragma: no cover
    app()
