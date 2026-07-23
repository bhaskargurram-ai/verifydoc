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


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0e8a16">
<title>VerifyDoc</title>
<style>
 :root{--ok:#0e8a16;--rev:#d93f0b;--bg:#0d1117;--fg:#e6edf3;--card:#161b22;--mut:#8b949e}
 *{box-sizing:border-box} body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
 header{padding:20px;border-bottom:1px solid #30363d} h1{margin:0;font-size:20px}
 .sub{color:var(--mut);font-size:13px} main{max-width:820px;margin:0 auto;padding:20px}
 label{display:block;margin:12px 0 4px;font-size:13px;color:var(--mut)}
 textarea,input,select{width:100%;padding:8px;background:var(--card);color:var(--fg);border:1px solid #30363d;border-radius:6px;font:inherit}
 textarea{min-height:80px} .row{display:flex;gap:12px} .row>div{flex:1}
 button{margin-top:14px;padding:10px 16px;background:var(--ok);color:#fff;border:0;border-radius:6px;font-weight:600;cursor:pointer}
 .field{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;margin:6px 0;background:var(--card);border-left:4px solid var(--mut);border-radius:6px}
 .accept{border-left-color:var(--ok)} .review{border-left-color:var(--rev)}
 .val{font-weight:600} .meta{color:var(--mut);font-size:12px} .err{color:var(--rev)}
</style></head><body>
<header><h1>🔒 VerifyDoc</h1><div class="sub">Per-field confidence · source grounding · accept/review. Runs on your infra.</div></header>
<main>
 <label>Document (upload a PDF/image, or paste text below)</label>
 <input type="file" id="file">
 <label>…or paste document text</label>
 <textarea id="text" placeholder="Invoice #: INV-1&#10;Total: $1,234.50"></textarea>
 <div class="row">
  <div><label>Schema (JSON)</label><textarea id="schema">{"type":"object","properties":{"total":{"type":"number","x-numeric-tol":0.01}}}</textarea></div>
 </div>
 <div class="row">
  <div><label>Adapter</label><select id="adapter"><option>text-search</option><option>rapidocr</option><option>api-vlm</option></select></div>
  <div><label>Threshold</label><input id="threshold" type="number" value="0.8" step="0.05" min="0" max="1"></div>
 </div>
 <button onclick="run()">Verify</button>
 <div id="out"></div>
</main>
<script>
async function run(){
 const out=document.getElementById('out'); out.innerHTML='Verifying…';
 const schema=document.getElementById('schema').value;
 const adapter=document.getElementById('adapter').value;
 const threshold=document.getElementById('threshold').value;
 const file=document.getElementById('file').files[0];
 const text=document.getElementById('text').value;
 try{
  let res;
  if(file){
   const fd=new FormData(); fd.append('file',file); fd.append('schema',schema);
   fd.append('adapter',adapter); fd.append('threshold',threshold);
   res=await fetch('/verify/upload',{method:'POST',body:fd});
  }else{
   res=await fetch('/verify',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({document:text,schema:JSON.parse(schema),adapter,threshold:parseFloat(threshold)})});
  }
  if(!res.ok){out.innerHTML='<p class="err">'+(await res.text())+'</p>';return;}
  const data=await res.json(); render(data,out);
 }catch(e){out.innerHTML='<p class="err">'+e+'</p>';}
}
function render(data,out){
 const flat=[]; (function walk(o,p){for(const k in o){const v=o[k];const np=p?p+'.'+k:k;
   if(v&&typeof v==='object'&&'decision' in v)flat.push([np,v]);else if(v&&typeof v==='object')walk(v,np);}})(data.fields,'');
 let h='<p class="meta">'+data.n_accepted+' accepted, '+data.n_review+' to review</p>';
 for(const [path,f] of flat){
  h+='<div class="field '+f.decision+'"><div><div class="val">'+path+' = '+JSON.stringify(f.value)+'</div>'+
     '<div class="meta">confidence '+f.confidence.toFixed(2)+(f.grounding?(' · page '+f.grounding.page):' · not grounded')+'</div></div>'+
     '<div>'+(f.decision==='accept'?'✅':'⚠️')+'</div></div>';
 }
 out.innerHTML=h;
}
</script></body></html>"""


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    host = os.environ.get("VERIFYDOC_HOST", "0.0.0.0")  # noqa: S104 - containerized server
    port = int(os.environ.get("VERIFYDOC_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
