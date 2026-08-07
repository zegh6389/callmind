from __future__ import annotations

from ..config import Settings
from .base import CallStart, CallStop, MediaChunk, TelephonyAdapter, TelephonyEvent
from .telnyx import TelnyxAdapter
from .twilio import TwilioAdapter


def create_adapter(settings: Settings) -> TelephonyAdapter:
    provider = settings.telephony_provider.lower()
    if provider == "telnyx":
        return TelnyxAdapter()
    if provider == "twilio":
        return TwilioAdapter()
    raise ValueError(f"unknown telephony provider: {provider}")


__all__ = [
    "CallStart",
    "CallStop",
    "MediaChunk",
    "TelephonyAdapter",
    "TelephonyEvent",
    "TelnyxAdapter",
    "TwilioAdapter",
    "create_adapter",
]
