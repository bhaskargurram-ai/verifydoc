"""VerifyDoc self-hosted server: REST API + web review UI + messaging webhooks.

    pip install 'verifydoc[server]'
    verifydoc-server                 # uvicorn on :8000 (or `docker compose up`)

Everything runs on your own infrastructure; with a local adapter (text-search,
rapidocr, a local HF VLM) documents never leave the machine. Routes:
``GET /`` (web app), ``POST /verify``, ``POST /verify/upload``, ``GET /adapters``,
``GET /health``, and ``/webhooks/{telegram,whatsapp}``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from verifydoc import __version__, verify
from verifydoc.adapters import _REGISTRY, get_adapter
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.ingest import document_from_text, ingest_path


class VerifyRequest(BaseModel):
    """JSON body for ``POST /verify`` (``schema`` is aliased to avoid shadowing)."""

    model_config = ConfigDict(populate_by_name=True)

    document: str
    schema_def: dict[str, Any] = Field(alias="schema")
    adapter: str = "text-search"
    k: int = 1
    threshold: float = 0.8


def _resolve_adapter(name: str) -> ExtractorAdapter | None:
    """None -> pipeline default (text-search); otherwise the named adapter."""
    if name in ("", "text-search"):
        return None
    try:
        return get_adapter(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def build_app() -> FastAPI:
    app = FastAPI(
        title="VerifyDoc",
        version=__version__,
        description="Trust layer for document -> structured-JSON extraction.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/adapters")
    def adapters() -> dict[str, list[str]]:
        return {"adapters": sorted(_REGISTRY)}

    @app.post("/verify")
    def verify_endpoint(req: VerifyRequest) -> dict[str, Any]:
        doc = document_from_text("api", [req.document])
        result = verify(
            doc,
            req.schema_def,
            adapter=_resolve_adapter(req.adapter),
            k=req.k,
            threshold=req.threshold,
        )
        return result.to_dict()

    @app.post("/verify/upload")
    async def verify_upload(
        file: Annotated[UploadFile, File()],
        schema: Annotated[str, Form()],
        adapter: Annotated[str, Form()] = "text-search",
        k: Annotated[int, Form()] = 1,
        threshold: Annotated[float, Form()] = 0.8,
    ) -> dict[str, Any]:
        try:
            schema_def = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"schema is not valid JSON: {exc}") from exc
        data = await file.read()
        suffix = Path(file.filename or "upload").suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            doc = ingest_path(tmp_path)
            result = verify(
                doc, schema_def, adapter=_resolve_adapter(adapter), k=k, threshold=threshold
            )
            return result.to_dict()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    from verifydoc.server.webhooks import router as webhook_router

    app.include_router(webhook_router)
    return app


_INDEX_HTML = (Path(__file__).parent / "review.html").read_text(encoding="utf-8")


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    host = os.environ.get("VERIFYDOC_HOST", "0.0.0.0")  # noqa: S104 - containerized server
    port = int(os.environ.get("VERIFYDOC_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
