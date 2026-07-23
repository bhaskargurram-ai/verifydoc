"""Offline tests for the FastAPI server + messaging webhooks (no network, no creds).

Skips cleanly if the ``server`` extra isn't installed."""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from verifydoc.server.app import build_app  # noqa: E402
from verifydoc.server.poll import reply_for_message  # noqa: E402
from verifydoc.server.webhooks import (  # noqa: E402
    _bot_adapter,
    _iter_whatsapp_texts,
    format_verification_reply,
    handle_telegram_update,
    verify_document_file,
    verify_document_text,
    verify_whatsapp_signature,
)


@pytest.fixture(autouse=True)
def _force_local_extractor(monkeypatch):
    """Keep bot-helper tests offline regardless of the developer's shell env."""
    monkeypatch.setenv("VERIFYDOC_BOT_ADAPTER", "text-search")


RECEIPT = "Corner Cafe\nDate: 2024-05-01\nTotal: 7.70\n"
SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}


@pytest.fixture
def client():
    return TestClient(build_app())


class TestRestApi:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    def test_adapters_lists_registry(self, client):
        names = client.get("/adapters").json()["adapters"]
        assert "text-search" in names and "mock" in names

    def test_verify_text(self, client):
        r = client.post("/verify", json={"document": RECEIPT, "schema": SCHEMA, "threshold": 0.8})
        assert r.status_code == 200
        body = r.json()
        assert "fields" in body and body["n_accepted"] + body["n_review"] >= 1

    def test_verify_bad_adapter_400(self, client):
        r = client.post("/verify", json={"document": RECEIPT, "schema": SCHEMA, "adapter": "nope"})
        assert r.status_code == 400

    def test_verify_upload(self, client):
        files = {"file": ("receipt.txt", RECEIPT, "text/plain")}
        r = client.post("/verify/upload", files=files, data={"schema": json.dumps(SCHEMA)})
        assert r.status_code == 200 and "fields" in r.json()

    def test_verify_upload_bad_schema_400(self, client):
        files = {"file": ("r.txt", RECEIPT, "text/plain")}
        r = client.post("/verify/upload", files=files, data={"schema": "{not json"})
        assert r.status_code == 400

    def test_index_serves_web_app(self, client):
        r = client.get("/")
        assert (
            r.status_code == 200
            and "VerifyDoc" in r.text
            and "text/html" in r.headers["content-type"]
        )


class TestWebhookHelpers:
    def test_verify_document_text(self):
        result = verify_document_text(RECEIPT)
        assert any(f.path == "total" for f in result.fields)

    def test_format_reply_marks_decisions(self):
        reply = format_verification_reply(verify_document_text(RECEIPT))
        assert "Verified" in reply and ("✅" in reply or "⚠️" in reply)

    def test_whatsapp_signature_roundtrip(self):
        import hashlib
        import hmac

        body = b'{"x":1}'
        secret = "s3cret"
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert verify_whatsapp_signature(body, sig, secret) is True
        assert verify_whatsapp_signature(body, "sha256=bad", secret) is False
        assert verify_whatsapp_signature(body, None, secret) is False

    def test_handle_telegram_text(self):
        reply = handle_telegram_update({"message": {"text": RECEIPT, "chat": {"id": 1}}})
        assert reply is not None and "Verified" in reply

    def test_handle_telegram_no_text(self):
        assert handle_telegram_update({"message": {"sticker": {}}}) is None

    def test_iter_whatsapp_texts(self):
        payload = {
            "entry": [
                {"changes": [{"value": {"messages": [{"from": "123", "text": {"body": RECEIPT}}]}}]}
            ]
        }
        assert _iter_whatsapp_texts(payload) == [(RECEIPT, "123")]


class TestWebhookEndpoints:
    def test_telegram_ok_without_token(self, client):
        # no TELEGRAM_BOT_TOKEN in env -> parses + verifies but does not send; returns ok
        r = client.post(
            "/webhooks/telegram", json={"message": {"text": RECEIPT, "chat": {"id": 1}}}
        )
        assert r.status_code == 200 and r.json()["ok"] is True

    def test_whatsapp_verify_challenge(self, client, monkeypatch):
        monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "tok")
        r = client.get(
            "/webhooks/whatsapp",
            params={"hub.mode": "subscribe", "hub.verify_token": "tok", "hub.challenge": "42"},
        )
        assert r.status_code == 200 and r.text == "42"

    def test_whatsapp_verify_rejects_bad_token(self, client, monkeypatch):
        monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "tok")
        r = client.get(
            "/webhooks/whatsapp",
            params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"},
        )
        assert r.status_code == 403


class TestBotExtractorSelection:
    def test_default_is_local_text_search(self, monkeypatch):
        monkeypatch.delenv("VERIFYDOC_BOT_ADAPTER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _bot_adapter() is None  # pipeline default = text-search (private)

    def test_api_vlm_when_key_present(self, monkeypatch):
        monkeypatch.delenv("VERIFYDOC_BOT_ADAPTER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
        adapter = _bot_adapter()  # constructed only; no network until extract()
        assert adapter is not None and adapter.name == "api-vlm"

    def test_explicit_override(self, monkeypatch):
        monkeypatch.setenv("VERIFYDOC_BOT_ADAPTER", "text-search")
        assert _bot_adapter() is None

    def test_verify_document_file(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text(RECEIPT)
        result = verify_document_file(str(p))
        assert len(result.fields) >= 1  # text-search finds date/total in the receipt


class TestReplyForMessage:
    def test_text_message(self):
        assert "Verified" in reply_for_message({"text": RECEIPT})

    def test_document_with_injected_download(self, tmp_path):
        p = tmp_path / "r.txt"
        p.write_text(RECEIPT)
        reply = reply_for_message({"document": {"file_id": "x"}}, download=lambda fid: str(p))
        assert "Verified" in reply

    def test_no_content_prompts_user(self):
        assert "Send me" in reply_for_message({"sticker": {}})
