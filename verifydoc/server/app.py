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
    api_key: str | None = None  # bring-your-own key for api-vlm; per-request, never stored


# Adapters that keep documents on-box and need no API key — safe for a public demo.
_DEMO_ADAPTERS = {"", "text-search", "rapidocr"}


def _resolve_adapter(name: str, api_key: str | None = None) -> ExtractorAdapter | None:
    """None -> pipeline default (text-search); otherwise the named adapter.

    Bring-your-own-key: if the request supplies an ``api_key`` for ``api-vlm``,
    the adapter is built with that key (used for this request only, never stored)
    — this is how a public demo offers the paid model without a shared key.

    In demo mode (``VERIFYDOC_DEMO`` set), only local keyless adapters are allowed
    unless the user brings their own key, so a public instance can't be driven to
    a shared paid API or leak documents off-box.
    """
    if name == "api-vlm" and api_key:
        from verifydoc.adapters.api_vlm import AnthropicClient, APIVLMAdapter

        return APIVLMAdapter(client=AnthropicClient(api_key=api_key))
    if os.environ.get("VERIFYDOC_DEMO") and name not in _DEMO_ADAPTERS:
        raise HTTPException(
            status_code=400,
            detail="demo mode: use a local adapter, or select api-vlm and paste your own "
            "Anthropic API key (used only for this request, never stored).",
        )
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

    # CORS so a static front-end (e.g. a Hugging Face static Space) can call this
    # API cross-origin. The demo uses no cookies/credentials — only a per-request
    # BYO key in the body — so a wildcard origin is safe. Override the allow-list
    # with VERIFYDOC_CORS_ORIGINS (comma-separated) to lock it down.
    from fastapi.middleware.cors import CORSMiddleware

    origins_env = os.environ.get("VERIFYDOC_CORS_ORIGINS", "*").strip()
    allow_origins = (
        ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
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
            adapter=_resolve_adapter(req.adapter, req.api_key),
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
        api_key: Annotated[str | None, Form()] = None,
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
                doc,
                schema_def,
                adapter=_resolve_adapter(adapter, api_key),
                k=k,
                threshold=threshold,
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
    # honor Cloud Run / PaaS $PORT, else VERIFYDOC_PORT, else 8000
    port = int(os.environ.get("PORT") or os.environ.get("VERIFYDOC_PORT") or "8000")
    uvicorn.run(build_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
