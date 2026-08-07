from __future__ import annotations

import logging
import re
from datetime import UTC, date, timedelta
from typing import ClassVar

from .account import AccountTool
from .base import Tool, ToolResult
from .booking import BookingTool

log = logging.getLogger("callmind.tools.router")

_PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{5,}\d)")
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(
    r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _extract_phone(text: str) -> str | None:
    m = _PHONE_RE.search(text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    if not 10 <= len(digits) <= 15:
        return None
    return digits


def _extract_time(text: str) -> str | None:
    m = _TIME_RE.search(text)
    if not m:
        return None
    h = int(m.group(1))
    minutes = int(m.group(2)) if m.group(2) else 0
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    if h > 23 or minutes > 59:
        return None
    return f"{h:02d}:{minutes:02d}"


def _extract_date(text: str, today: date | None = None) -> str | None:
    from datetime import datetime

    today = today or datetime.now(UTC).date()
    m = _DATE_TOKEN_RE.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    if token == "today":
        return today.isoformat()
    if token == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    # Weekday name -> next occurrence
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if token in days:
        target = days.index(token)
        delta = (target - today.weekday()) % 7
        if delta == 0:
            delta = 7
        return (today + timedelta(days=delta)).isoformat()
    return None


class ToolRouter:
    """Whitelisted, deterministic tool dispatch.

    Maps an intent label to a single Tool. Never invokes tools the intent
    doesn't map to. Real CRM/Calendar integrations slot into BookingTool /
    AccountTool.run without changing this router.
    """

    WHITELIST: ClassVar[dict[str, type[Tool]]] = {
        "booking": BookingTool,
        "account_status": AccountTool,
    }

    def __init__(self, tools: dict[str, Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = tools or {
            intent: cls() for intent, cls in self.WHITELIST.items()
        }

    def available_tools(self) -> list[str]:
        return sorted(self._tools)

    async def dispatch(
        self,
        intent: str,
        params: dict,
        *,
        call_id: str,
        business_id: str,
    ) -> ToolResult | None:
        tool = self._tools.get(intent)
        if tool is None:
            log.debug("no tool for intent=%s", intent)
            return None
        return await tool.run(params, call_id=call_id, business_id=business_id)

    def extract_params(self, intent: str, user_text: str) -> dict:
        if intent == "booking":
            return {
                "date": _extract_date(user_text),
                "time": _extract_time(user_text),
                "title": _extract_title(user_text),
                "caller_name": _extract_name(user_text),
            }
        if intent == "account_status":
            return {"caller_phone": _extract_phone(user_text)}
        return {}


def _extract_title(text: str) -> str:
    m = re.search(
        r"(?:book(?:ing)?|schedule)\s+(?:a |an )?(.+?)(?:\s+(?:on|at|for|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\?|$)",
        text,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_name(text: str) -> str:
    m = re.search(
        r"(?:this is|i'm|im|my name is|name's|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # fallback: first capitalized 1-2 word token
    m = re.search(r"\b([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)\b", text)
    return m.group(1).strip() if m else ""