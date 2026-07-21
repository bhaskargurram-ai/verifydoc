"""Ingest stage: turn a source file into a typed Document (pages + text layer)."""

from verifydoc.ingest.loader import document_from_text, ingest_path

__all__ = ["document_from_text", "ingest_path"]
