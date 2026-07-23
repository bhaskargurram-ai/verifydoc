"""CLI: ``verifydoc extract doc.pdf --schema schema.json --target... ``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

import verifydoc
from verifydoc.adapters import get_adapter
from verifydoc.pipeline import DEFAULT_THRESHOLD, verify, verify_batch

app = typer.Typer(name="verifydoc", help="Trust layer for document -> JSON extraction.")

# Default file types picked up by ``batch`` when no explicit --glob is given.
_DOC_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".txt", ".md"}


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
def batch(
    directory: Path = typer.Argument(..., help="Folder of documents (pdf/image/txt)."),
    schema: Path = typer.Option(..., "--schema", "-s", help="JSON Schema with x-scoring rules."),
    out_dir: Path = typer.Option(
        Path("verifydoc_out"), "--out-dir", "-o", help="Directory for per-doc JSON + summary."
    ),
    adapter: str = typer.Option("text-search", "--adapter", "-a", help="Extractor adapter."),
    k: int = typer.Option(1, "--k", help="Self-consistency samples (k > 1 enables consensus)."),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD, "--threshold", "-t", help="Accept-decision confidence cutoff."
    ),
    glob: str = typer.Option(
        "*", "--glob", "-g", help="Filename pattern; default keeps known document types."
    ),
) -> None:
    """Verify every document in a folder against one schema.

    Writes one ``<stem>.json`` per document plus a ``summary.json`` with the
    per-doc and total accepted/review counts, into ``--out-dir``.
    """
    files = [p for p in sorted(directory.glob(glob)) if p.is_file()]
    if glob == "*":
        files = [p for p in files if p.suffix.lower() in _DOC_EXTS]
    if not files:
        typer.echo(f"no documents matching {glob!r} in {directory}", err=True)
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = verify_batch(files, schema, adapter=get_adapter(adapter), k=k, threshold=threshold)

    rows = []
    for src, res in zip(files, results):
        dest = out_dir / f"{src.stem}.json"
        dest.write_text(
            json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "file": src.name,
                "doc_id": res.doc_id,
                "n_fields": len(res.fields),
                "n_accepted": res.n_accepted,
                "n_review": res.n_review,
                "output": dest.name,
            }
        )
        typer.echo(
            f"{src.name}: {res.n_accepted} accepted, {res.n_review} to review -> {dest.name}"
        )

    summary = {
        "directory": str(directory),
        "adapter": adapter,
        "threshold": threshold,
        "n_docs": len(results),
        "total_accepted": sum(r.n_accepted for r in results),
        "total_review": sum(r.n_review for r in results),
        "documents": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"\n{len(results)} docs -> {out_dir}/ "
        f"({summary['total_accepted']} accepted, {summary['total_review']} to review); "
        "summary.json written"
    )


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
