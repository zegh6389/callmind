from __future__ import annotations

import base64
import binascii
import json
import logging

from .base import CallStart, CallStop, MediaChunk, TelephonyAdapter, TelephonyEvent

log = logging.getLogger("callmind.telephony.telnyx")


class TelnyxAdapter(TelephonyAdapter):
    """Telnyx Media Streaming over WebSockets (PCMU 8kHz, base64 RTP payload).

    Schema per developers.telnyx.com/docs/voice/programmable-voice/media-streaming:
    inbound events: connected / start / media / stop / dtmf / mark / error.
    outbound: media (bidirectional rtp mode), clear, mark.
    """

    name = "telnyx"

    def __init__(self) -> None:
        self._stream_id: str | None = None

    def parse(self, message: str | bytes) -> TelephonyEvent | None:
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            log.warning("unparseable telephony message: %r", message[:200])
            return None

        event = data.get("event")
        if event == "connected":
            return None
        if event == "start":
            start = data.get("start", {})
            self._stream_id = data.get("stream_id")
            return CallStart(
                call_id=start.get("call_control_id") or self._stream_id or "",
                stream_id=self._stream_id,
                from_number=start.get("from"),
                to_number=start.get("to"),
                raw=data,
            )
        if event == "media":
            media = data.get("media", {})
            track = media.get("track", "inbound")
            if track != "inbound":
                return None
            payload = media.get("payload", "")
            if not payload:
                return None
            seq_raw = media.get("chunk")
            try:
                raw = base64.b64decode(payload, validate=False)
            except (binascii.Error, ValueError):
                log.warning("telnyx: malformed media payload")
                return None
            return MediaChunk(
                payload=raw,
                stream_id=data.get("stream_id"),
                seq=int(seq_raw) if seq_raw else None,
                track=track,
            )
        if event == "stop":
            stop = data.get("stop", {})
            return CallStop(call_id=stop.get("call_control_id"))
        if event == "error":
            log.error("telnyx stream error: %s", data.get("payload"))
            return None
        if event in ("dtmf", "mark"):
            return None
        return None

    def media_message(self, mulaw_bytes: bytes, seq: int) -> dict:
        return {
            "event": "media",
            "media": {"payload": base64.b64encode(mulaw_bytes).decode("ascii")},
        }

    def clear_message(self) -> dict | None:
        return {"event": "clear"}

    def start_message(self) -> dict | None:
        """No client->server handshake required.

        Telnyx pushes connected+start events to the WebSocket server;
        bidirectional RTP settings travel on the answer API call, so the
        client does not negotiate any setup frame.
        """
        return None
