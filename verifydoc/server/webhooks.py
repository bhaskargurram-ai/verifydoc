"""Messaging webhooks: send a document to a Telegram / WhatsApp bot, get back
VerifyDoc's per-field trust report.

The pure logic — signature verification, update parsing, verify+format — lives
in helpers that are unit-tested offline with no credentials. The network I/O
(download media, send reply) needs live tokens, reads them from the environment
(never committed), and is marked ``# pragma: no cover``.

Privacy: with a local adapter the document is verified on your server. The
Telegram path can be fully self-hosted; WhatsApp Cloud API necessarily routes
media through Meta — documented in docs/DEPLOY.md.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from verifydoc import verify
from verifydoc.ingest import document_from_text
from verifydoc.pipeline import VerifiedResult

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# A generic schema for ad-hoc bot use (receipts / invoices). Callers wanting a
# specific schema should use the REST API, which takes one per request.
DEFAULT_BOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "x-scoring": "semantic"},
        "date": {"type": "string"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


def format_verification_reply(result: VerifiedResult) -> str:
    """Human-readable chat reply: each field with an accept/review marker."""
    header = (
        f"Verified {len(result.fields)} field(s) — "
        f"{result.n_accepted} accepted, {result.n_review} to review:"
    )
    lines = [header]
    for f in result.fields:
        mark = "✅" if f.decision == "accept" else "⚠️"
        lines.append(f"{mark} {f.path}: {f.value!r} (conf {f.confidence:.2f})")
    if result.n_review:
        lines.append("\n⚠️ fields need a human — check them against the source.")
    return "\n".join(lines)


def verify_whatsapp_signature(body: bytes, signature: str | None, app_secret: str) -> bool:
    """Constant-time check of Meta's ``X-Hub-Signature-256`` header."""
    expected = "sha256=" + hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def verify_document_text(text: str, schema: dict[str, Any] | None = None) -> VerifiedResult:
    """Run the trust layer on a plain-text document (local text-search adapter)."""
    return verify(document_from_text("bot", [text]), schema or DEFAULT_BOT_SCHEMA, threshold=0.8)


def handle_telegram_update(update: dict[str, Any]) -> str | None:
    """Parse a Telegram update; return the reply text, or None if nothing to do.

    Handles text messages here (offline-testable). Document/photo attachments
    require a download round-trip and are dispatched by the endpoint.
    """
    message = update.get("message") or update.get("edited_message") or {}
    text = message.get("text")
    if not text:
        return None
    return format_verification_reply(verify_document_text(text))


@router.post("/telegram")
async def telegram_webhook(request: Request) -> JSONResponse:
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if secret and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
        raise HTTPException(status_code=403, detail="invalid webhook secret")
    update = await request.json()
    reply = handle_telegram_update(update)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if reply and token:  # pragma: no cover - network
        await _telegram_send(token, update["message"]["chat"]["id"], reply)
    return JSONResponse({"ok": True})


@router.get("/whatsapp")
def whatsapp_verify(request: Request) -> PlainTextResponse:
    """Meta webhook verification handshake (echo hub.challenge)."""
    params = request.query_params
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN")
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == verify_token:
        return PlainTextResponse(params.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="verification failed")


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request) -> JSONResponse:
    app_secret = os.environ.get("WHATSAPP_APP_SECRET")
    body = await request.body()
    if app_secret and not verify_whatsapp_signature(
        body, request.headers.get("X-Hub-Signature-256"), app_secret
    ):
        raise HTTPException(status_code=403, detail="invalid signature")
    payload = await request.json()
    for text, sender in _iter_whatsapp_texts(payload):
        reply = format_verification_reply(verify_document_text(text))
        token = os.environ.get("WHATSAPP_TOKEN")
        phone_id = os.environ.get("WHATSAPP_PHONE_ID")
        if token and phone_id:  # pragma: no cover - network
            await _whatsapp_send(token, phone_id, sender, reply)
    return JSONResponse({"ok": True})


def _iter_whatsapp_texts(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ``(text, sender)`` pairs from a WhatsApp Cloud API payload."""
    out: list[tuple[str, str]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []):
                text = (msg.get("text") or {}).get("body")
                if text:
                    out.append((text, msg.get("from", "")))
    return out


async def _telegram_send(token: str, chat_id: int, text: str) -> None:  # pragma: no cover - network
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def _whatsapp_send(
    token: str, phone_id: str, to: str, text: str
) -> None:  # pragma: no cover - network
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"https://graph.facebook.com/v20.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": to, "text": {"body": text}},
        )
