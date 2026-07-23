# Integration quickstart

VerifyDoc adds a per-field trust layer (calibrated confidence + source grounding
+ accept/review) on top of whatever extractor or framework you already use. None
of these frameworks is a hard dependency — the integrations work by duck typing.

## Instructor / Outlines / Marvin / Pydantic-AI (any `BaseModel`)

Verify an object that any structured-extraction framework produced:

```python
from verifydoc.integrations.instructor import verify_instructor_result

obj = my_framework_extract(document_text)      # -> a pydantic.BaseModel instance
result = verify_instructor_result(document_text, obj, threshold=0.8)

for f in result.fields:
    print(f.path, f.value, round(f.confidence, 2), f.decision)   # accept / review
```

## Pydantic-AI (or plain Pydantic)

```python
from pydantic import BaseModel
from verifydoc import verify_model

class Invoice(BaseModel):
    invoice_id: str
    total: float

result = verify_model("invoice.pdf", Invoice, threshold=0.8)
print(result.to_dict())
```

## LangChain

```python
from verifydoc.integrations.langchain import VerifiedExtractor

extractor = VerifiedExtractor(chain.invoke, schema=Invoice, threshold=0.8)
result = extractor(document_text)              # -> VerifiedResult
```

## LlamaIndex / DSPy / Haystack

Wrap your `str -> dict | BaseModel` extraction step and hand the output to
`verify_instructor_result` (BaseModel) or `verify(text, schema=...)` (dict). See
runnable scripts in [`examples/`](../examples/):
- `examples/pydantic_ai_example.py`
- `examples/llamaindex_example.py`

## Agents / IDEs (MCP)

Claude Code, Claude Desktop, Cursor, Cline, Codex — point them at the
`verifydoc-mcp` stdio server. Copy-paste configs: [`examples/mcp/README.md`](../examples/mcp/README.md).

## Self-hosted API / web / messaging

Run the FastAPI server for a REST `/verify` endpoint, a review web app, and
WhatsApp/Telegram bots — all on your own infra. See [`docs/DEPLOY.md`](DEPLOY.md).
