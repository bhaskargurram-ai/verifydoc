# Deploy VerifyDoc: self-hosted API, web app, and messaging bots

VerifyDoc is designed to be **self-hosted**, so your documents stay on your own
infrastructure. Three surfaces share one backend:

- **REST API** — `POST /verify` (text+schema) and `POST /verify/upload` (file),
  plus `/adapters` and `/health`.
- **Web app** — upload → review queue with grounding overlays → download verified
  JSON. Installable as a PWA on a phone home screen.
- **Messaging bots** — send a document to a **Telegram** or **WhatsApp** bot and
  get back the extracted fields with confidence + which ones need review.

## One command

```bash
docker compose up          # backend + web UI on http://localhost:8000
# or, from pip:
pip install 'verifydoc[server]'
verifydoc-server           # uvicorn app on :8000
```

## Run the Telegram bot with no public URL

Webhooks need a public HTTPS endpoint. To run the bot from a laptop or an
air-gapped box with **no open port**, use long-polling instead:

```bash
pip install 'verifydoc[server]'
export TELEGRAM_BOT_TOKEN=...            # from @BotFather
# optional — use Claude to extract fields (else the local text-search baseline):
export ANTHROPIC_API_KEY=...             # + pip install 'verifydoc[api]'
verifydoc-bot                            # polls Telegram; message the bot a receipt/PDF
```

The bot extractor is chosen by `VERIFYDOC_BOT_ADAPTER` (defaults to `api-vlm`
when an Anthropic key is present, else `text-search`). Set it to `rapidocr` for
local OCR on images.

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

Shipping now (`pip install 'verifydoc[server]'`): the FastAPI server + REST API,
the single-page web review UI, the Dockerfile/compose, the Telegram bot
(webhook **and** polling via `verifydoc-bot`), and the WhatsApp Cloud API webhook.
The library, CLI, MCP server, and Streamlit UI ship in the base package. A
richer multi-reviewer web queue and native mobile app remain on the roadmap.
