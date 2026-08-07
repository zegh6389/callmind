from __future__ import annotations

import base64
import binascii
import json
import logging

from .base import CallStart, CallStop, MediaChunk, TelephonyAdapter, TelephonyEvent

log = logging.getLogger("callmind.telephony.twilio")


class TwilioAdapter(TelephonyAdapter):
    """Twilio Media Streams protocol (start/media/dtmf/stop, base64 mu-law 8kHz)."""

    name = "twilio"

    def __init__(self) -> None:
        self._stream_sid: str | None = None

    def parse(self, message: str | bytes) -> TelephonyEvent | None:
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            log.warning("unparseable telephony message: %r", message[:200])
            return None

        event = data.get("event")
        if event == "start":
            start = data.get("start", {})
            self._stream_sid = data.get("streamSid") or start.get("streamSid")
            return CallStart(
                call_id=start.get("callSid") or self._stream_sid or "",
                stream_id=self._stream_sid,
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
                log.warning("twilio: malformed media payload")
                return None
            return MediaChunk(
                payload=raw,
                stream_id=data.get("streamSid"),
                seq=int(seq_raw) if seq_raw else None,
                track=track,
            )
        if event == "stop":
            return CallStop(call_id=data.get("callSid"))
        if event == "dtmf":
            return None
        return None

    def media_message(self, mulaw_bytes: bytes, seq: int) -> dict:
        return {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": base64.b64encode(mulaw_bytes).decode("ascii")},
        }

    def clear_message(self) -> dict | None:
        if not self._stream_sid:
            return None
        return {"event": "clear", "streamSid": self._stream_sid}
