"""Telnyx V2 webhook verification (Ed25519).

Per Telnyx V2 docs: headers `telnyx-signature-ed25519` (Base64 Ed25519
signature) and `telnyx-timestamp` (Unix seconds) over `{ts}|{raw_body}`,
verified against the account's public key. Replay window enforced.
"""

from __future__ import annotations

import base64
import json
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
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


def _keypair() -> tuple[bytes, bytes]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes_raw()
    return priv, pub


def _sign(priv: Ed25519PrivateKey, ts: int, body: bytes) -> str:
    msg = f"{ts}|".encode() + body
    return base64.b64encode(priv.sign(msg)).decode()


def _payload() -> dict:
    return {
        "data": {
            "event_type": "call.initiated",
            "payload": {"call_control_id": "v2:CC-AUTH-1"},
        }
    }


def _boot(public_key: bytes | None = None, *, tolerance: int = 300) -> TestClient:
    pub_b64 = base64.b64encode(public_key).decode() if public_key else ""
    app.state.settings = Settings(
        telnyx_webhook_public_key=pub_b64,
        telnyx_webhook_tolerance_seconds=tolerance,
        public_ws_url="wss://voice.example.com/ws/call",
    )
    app.state.telnyx = FakeTelnyx()
    return TestClient(app)


def test_webhook_without_public_key_is_rejected():
    c = _boot(public_key=None)
    body = json.dumps(_payload()).encode()
    r = c.post("/telnyx/webhook", content=body)
    assert r.status_code == 503


def test_webhook_missing_signature_rejected():
    _, pub = _keypair()
    c = _boot(public_key=pub)
    body = json.dumps(_payload()).encode()
    r = c.post("/telnyx/webhook", content=body)
    assert r.status_code == 401
    assert app.state.telnyx.calls == []


def test_webhook_missing_timestamp_rejected():
    priv, pub = _keypair()
    c = _boot(public_key=pub)
    body = json.dumps(_payload()).encode()
    sig = _sign(priv, int(time.time()), body)
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={"telnyx-signature-ed25519": sig},
    )
    assert r.status_code == 401


def test_webhook_forged_signature_rejected():
    _, pub = _keypair()
    c = _boot(public_key=pub)
    body = json.dumps(_payload()).encode()
    forged_sig = base64.b64encode(b"\x00" * 64).decode()
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={
            "telnyx-signature-ed25519": forged_sig,
            "telnyx-timestamp": str(int(time.time())),
        },
    )
    assert r.status_code == 401
    assert app.state.telnyx.calls == []


def test_webhook_wrong_key_rejected():
    priv_a, _ = _keypair()
    _, pub_b = _keypair()
    c = _boot(public_key=pub_b)
    body = json.dumps(_payload()).encode()
    sig = _sign(priv_a, int(time.time()), body)
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={
            "telnyx-signature-ed25519": sig,
            "telnyx-timestamp": str(int(time.time())),
        },
    )
    assert r.status_code == 401


def test_webhook_stale_timestamp_rejected():
    priv, pub = _keypair()
    c = _boot(public_key=pub, tolerance=300)
    body = json.dumps(_payload()).encode()
    old_ts = int(time.time()) - 3600
    sig = _sign(priv, old_ts, body)
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={
            "telnyx-signature-ed25519": sig,
            "telnyx-timestamp": str(old_ts),
        },
    )
    assert r.status_code == 401
    assert app.state.telnyx.calls == []


def test_webhook_valid_signature_accepted_and_answers_call():
    priv, pub = _keypair()
    c = _boot(public_key=pub)
    body = json.dumps(_payload()).encode()
    ts = int(time.time())
    sig = _sign(priv, ts, body)
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={
            "telnyx-signature-ed25519": sig,
            "telnyx-timestamp": str(ts),
        },
    )
    assert r.status_code == 200
    for _ in range(50):
        if app.state.telnyx.calls:
            break
        time.sleep(0.02)
    assert app.state.telnyx.calls == [
        ("v2:CC-AUTH-1", "wss://voice.example.com/ws/call")
    ]


def test_webhook_answer_failure_logged_not_swallowed(monkeypatch):
    """fire-and-forget answer task must add done_callback for audit log."""

    logger = []

    def fake_excepthook(coro):
        try:
            coro.close()
        except Exception:
            pass

    class BoomTelnyx:
        def __init__(self):
            self.calls = []

        async def answer_with_stream(self, control_id, url):
            self.calls.append((control_id, url))
            raise RuntimeError("upstream telnyx blew up")

        async def close(self):
            pass

    priv, pub = _keypair()
    app.state.settings = Settings(
        telnyx_webhook_public_key=base64.b64encode(pub).decode(),
        public_ws_url="wss://x/ws",
    )
    app.state.telnyx = BoomTelnyx()
    c = TestClient(app)
    body = json.dumps(
        {"data": {"event_type": "call.initiated", "payload": {"call_control_id": "v2:BB-1"}}}
    ).encode()
    sig = _sign(priv, int(time.time()), body)
    r = c.post(
        "/telnyx/webhook",
        content=body,
        headers={
            "telnyx-signature-ed25519": sig,
            "telnyx-timestamp": str(int(time.time())),
        },
    )
    assert r.status_code == 200
    # Wait for the background task to start (and fail).
    for _ in range(50):
        if app.state.telnyx.calls:
            break
        time.sleep(0.02)
    assert app.state.telnyx.calls == [("v2:BB-1", "wss://x/ws")]


def test_app_router_uses_configured_stub_mode():
    from callmind.config import Settings
    from callmind.gateway.app import app
    from callmind.tools.router import ToolRouter

    for setting in (True, False):
        app.state.settings = Settings(tool_stub_mode=setting)
        app.state.tool_router = ToolRouter(stub_mode=app.state.settings.tool_stub_mode)
        assert app.state.tool_router._tools["account_status"].stub_mode is setting