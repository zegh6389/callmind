"""Telnyx webhook signature verification (F1).

Gateway must reject unsigned/forged Telnyx webhooks, else any internet
caller can POST call.initiated and make the gateway answer calls.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from callmind.config import Settings
from callmind.gateway.app import app


class FakeTelnyx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def answer_with_stream(self, call_control_id: str, stream_url: str) -> None:
        self.calls.append((call_control_id, stream_url))

    async def close(self) -> None:
        pass


def _sig(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payload() -> dict:
    return {"data": {"event_type": "call.initiated", "payload": {"call_control_id": "v2:CC-AUTH-1"}}}


def _boot(secret: str) -> TestClient:
    app.state.settings = Settings(
        telnyx_webhook_secret=secret,
        public_ws_url="wss://voice.example.com/ws/call",
    )
    app.state.telnyx = FakeTelnyx()
    return TestClient(app)


def test_webhook_without_secret_configured_is_rejected():
    c = _boot(secret="")
    body = json.dumps(_payload()).encode()
    r = c.post("/telnyx/webhook", content=body)
    assert r.status_code == 503


def test_webhook_missing_signature_rejected():
    c = _boot("sekrit")
    body = json.dumps(_payload()).encode()
    r = c.post("/telnyx/webhook", content=body)
    assert r.status_code == 401
    assert app.state.telnyx.calls == []


def test_webhook_forged_signature_rejected():
    c = _boot("sekrit")
    body = json.dumps(_payload()).encode()
    r = c.post("/telnyx/webhook", content=body, headers={"x-telnyx-signature": "deadbeef" * 8})
    assert r.status_code == 401
    assert app.state.telnyx.calls == []


def test_webhook_valid_signature_accepted_and_answers_call():
    c = _boot("sekrit")
    body = json.dumps(_payload()).encode()
    sig = _sig(body, "sekrit")
    r = c.post("/telnyx/webhook", content=body, headers={"x-telnyx-signature": sig})
    assert r.status_code == 200
    for _ in range(50):
        if app.state.telnyx.calls:
            break
        time.sleep(0.02)
    assert app.state.telnyx.calls == [("v2:CC-AUTH-1", "wss://voice.example.com/ws/call")]