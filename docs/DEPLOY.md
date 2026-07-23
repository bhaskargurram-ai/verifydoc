# Deploy VerifyDoc: self-hosted API, web app, and messaging bots

VerifyDoc is designed to be **self-hosted**, so your documents stay on your own
infrastructure. Three surfaces share one backend:

- **REST API** — `POST /verify` (text+schema) and `POST /verify/upload` (file),
  plus `/adapters` and `/health`.
- **Web app** — upload → review queue with grounding overlays → download verified
  JSON. Installable as a PWA on a phone home screen.
- **Messaging bots** — send a document to a **Telegram** or **WhatsApp** bot and
  get back the extracted fields with confidence + which ones need review.

## One command (planned v1.0)

```bash
docker compose up          # backend + web UI on http://localhost:8000
# or, from pip:
pip install 'verifydoc[server]'
verifydoc-server           # uvicorn app on :8000
```

## Privacy model

- With a **local** adapter (`text-search`, `rapidocr`, a local HF VLM) the
  document never leaves your server — end-to-end private.
- The **Telegram** bot can be fully self-hosted; media flows only between the
  user and your server.
- **WhatsApp Cloud API** necessarily routes media through Meta's servers — use it
  only when that is acceptable, or prefer Telegram / the web app for full privacy.
- Inbound webhooks are signature-verified; secrets are read from the environment
  (never committed).

## Status

The library, CLI, MCP server, and Streamlit review UI ship today
(`pip install verifydoc`). The FastAPI server, web app, Docker image, and
Telegram/WhatsApp bots are the **v1.0 self-host release** — tracked in the
project roadmap. This document is the design contract they implement.
