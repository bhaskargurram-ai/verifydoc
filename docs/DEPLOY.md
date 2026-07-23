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

## Deployment options

Pick by privacy needs and whether you need inbound webhooks. Every option runs
the same image/app; **local adapters keep documents on the box**, while `api-vlm`
or the WhatsApp Cloud API send content to a third party.

| Where | How | Best for | Privacy |
|---|---|---|---|
| Laptop / on-prem | `pip install 'verifydoc[server]'` → `verifydoc-server` | dev, sensitive docs | 100% local |
| Docker, one box | `docker run -p 8000:8000 ghcr.io/bhaskargurram-ai/verifydoc` · `docker compose up` | simple self-host | 100% local |
| Bot, no public URL | `verifydoc-bot` (Telegram long-poll) | quick bot behind NAT/firewall | local (media via Telegram) |
| Serverless containers | Cloud Run / AWS App Runner / Azure Container Apps — run the ghcr image | REST API + webhooks, scale-to-zero, free HTTPS | your cloud |
| PaaS one-click | Fly.io / Render / Railway (build from the Dockerfile) | fastest cloud deploy | your account |
| Cloud VM | EC2/GCE/Hetzner + Docker + Caddy or a Cloudflare Tunnel for HTTPS | full control | your infra |
| HF Spaces / Streamlit Cloud | the Streamlit UI as a public demo | showcase only — not private docs | public |
| GPU box | any 24 GB GPU VM (RunPod/Lambda/vast/cloud) + the image | local `hf-vlm` / PaddleOCR extraction | local |

Notes:

- **Webhooks (WhatsApp, Telegram-webhook) need a public HTTPS endpoint.**
  Serverless/PaaS give you one for free; on a VM use Caddy or a Cloudflare
  Tunnel. The **Telegram polling** runner (`verifydoc-bot`) needs no inbound
  connection at all.
- **Config via env vars** (all optional): `VERIFYDOC_HOST`/`VERIFYDOC_PORT`,
  `VERIFYDOC_BOT_ADAPTER`, `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`, and the
  Telegram/WhatsApp tokens (see below). Never bake secrets into the image —
  pass them at run time.
- **GPU** is only needed for the local VLM/heavy-OCR adapters (`hf-vlm`,
  `paddleocr-vl`); a single 24 GB card is enough. The default `text-search` /
  `rapidocr` adapters run on CPU.
- **Recommendation:** private teams → Docker/Compose on your own VM or GPU box;
  public demo → a Hugging Face Space; low-ops API + bot → Cloud Run.

## Hosted demo (Cloud Run + Firebase Hosting)

Run a public demo on **GCP Cloud Run** (which runs the container), optionally
fronted by **Firebase Hosting** for a `*.web.app` URL. Firebase Hosting *alone*
cannot run the Python server — it serves static files and rewrites `**` to the
Cloud Run service (see `firebase.json`).

Deploy it (needs a GCP project with billing + `gcloud`/`firebase` CLIs logged in):

```bash
# 1. deploy the container to Cloud Run (builds from the Dockerfile)
gcloud run deploy verifydoc --source . --region us-central1 \
  --allow-unauthenticated --set-env-vars VERIFYDOC_DEMO=1

# 2. (optional) front it with Firebase Hosting for a web.app URL
#    edit .firebaserc -> your project id (region in firebase.json must match), then:
firebase deploy --only hosting
```

Cloud Run injects `$PORT`, which the server honors. Step 1 alone gives a working
`https://verifydoc-….run.app`; step 2 maps it under `yourproject.web.app`.

**Demo safety:** deploy with `VERIFYDOC_DEMO=1` and **no API keys**. That
restricts extraction to local, keyless adapters (`text-search`, `rapidocr`), so a
public instance can't be driven to a paid API or made to send documents off-box.

**Hands-off CI/CD:** add a GCP service-account key + project id as GitHub secrets
and wire a `workflow_dispatch` job (`google-github-actions/deploy-cloudrun` +
`FirebaseExtended/action-hosting-deploy`).

**Simplest non-GCP demo:** push the same image to a **Hugging Face Space**
(Docker SDK) — a public URL with no cloud account beyond HF.

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
