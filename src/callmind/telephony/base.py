from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CallStart:
    call_id: str
    stream_id: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MediaChunk:
    payload: bytes
    stream_id: str | None = None
    seq: int | None = None
    track: str = "inbound"


@dataclass(frozen=True)
class CallStop:
    call_id: str | None = None


TelephonyEvent = CallStart | MediaChunk | CallStop


class TelephonyAdapter(ABC):
    """Provider-agnostic bridge between the gateway and a telephony WebSocket protocol."""

    name: str = "base"

    @abstractmethod
    def parse(self, message: str | bytes) -> TelephonyEvent | None:
        """Parse one inbound WebSocket message into a gateway event."""

    @abstractmethod
    def media_message(self, mulaw_bytes: bytes, seq: int) -> dict:
        """Build an outbound media frame message carrying 20ms of mu-law audio."""

    def clear_message(self) -> dict | None:
        """Optional message that flushes the provider's playback buffer (barge-in)."""
        return None

    def start_message(self) -> dict | None:
        """Optional message the client must send to open the media stream (Telnyx)."""
        return None
