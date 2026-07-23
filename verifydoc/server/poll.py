"""Run the Telegram bot by long-polling — no public webhook / open port needed.

    pip install 'verifydoc[server]'
    export TELEGRAM_BOT_TOKEN=...            # from @BotFather
    # optional: export ANTHROPIC_API_KEY=... # use Claude to extract (else local text-search)
    verifydoc-bot

Reuses verifydoc.server.webhooks (the same verify+format logic as the webhook
endpoint), so text -> verify_document_text and document/photo -> download +
verify_document_file. The network loop is thin and marked no-cover; the message
dispatch (:func:`reply_for_message`) is pure and unit-tested with an injected
downloader.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verifydoc.server.webhooks import (
    format_verification_reply,
    verify_document_file,
    verify_document_text,
)

Downloader = Callable[[str], str]


def reply_for_message(message: dict[str, Any], download: Downloader | None = None) -> str:
    """Build the bot's reply for one Telegram message.

    ``download(file_id) -> local_path`` is injected so the text path is fully
    offline-testable and documents/photos are handled when a downloader is given.
    """
    text = message.get("text")
    if text:
        return format_verification_reply(verify_document_text(text))
    file_id = None
    if message.get("document"):
        file_id = message["document"]["file_id"]
    elif message.get("photo"):
        file_id = message["photo"][-1]["file_id"]  # largest rendition
    if file_id and download is not None:
        return format_verification_reply(verify_document_file(download(file_id)))
    return "Send me a receipt/invoice as text, a photo, or a PDF and I'll verify each field."


def _make_downloader(token: str) -> Downloader:  # pragma: no cover - network
    def _download(file_id: str) -> str:
        meta_url = f"https://api.telegram.org/bot{token}/getFile?" + urllib.parse.urlencode(
            {"file_id": file_id}
        )
        with urllib.request.urlopen(meta_url, timeout=60) as r:
            file_path = json.load(r)["result"]["file_path"]
        suffix = Path(file_path).suffix or ".bin"
        file_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as dest:
            with urllib.request.urlopen(file_url, timeout=120) as r:
                dest.write(r.read())
            return dest.name

    return _download


def main() -> None:  # pragma: no cover - network entrypoint
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN is not set (get one from @BotFather).")
    base = f"https://api.telegram.org/bot{token}"
    download = _make_downloader(token)
    print("verifydoc-bot polling; press Ctrl+C to stop.", flush=True)
    offset = 0
    while True:
        try:
            url = f"{base}/getUpdates?" + urllib.parse.urlencode({"offset": offset, "timeout": 25})
            with urllib.request.urlopen(url, timeout=60) as r:
                updates = json.load(r).get("result", [])
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print("getUpdates error:", str(exc)[:150], flush=True)
            time.sleep(3)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue
            try:
                reply = reply_for_message(message, download)
            except Exception as exc:
                reply = f"verification error: {str(exc)[:180]}"
            send = f"{base}/sendMessage?" + urllib.parse.urlencode(
                {"chat_id": message["chat"]["id"], "text": reply}
            )
            try:
                urllib.request.urlopen(send, timeout=30).read()
            except Exception as exc:
                print("sendMessage error:", str(exc)[:150], flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
