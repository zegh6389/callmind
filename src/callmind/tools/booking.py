from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime

from .base import Tool, ToolResult

log = logging.getLogger("callmind.tools.booking")

REQUIRED = ("title", "date", "time", "caller_name")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")


class BookingTool(Tool):
    name = "booking.create_event"
    description = "Create a calendar event for the caller."

    async def run(
        self,
        params: dict,
        *,
        call_id: str,
        business_id: str,
    ) -> ToolResult:
        missing = [k for k in REQUIRED if not params.get(k)]
        if missing:
            return ToolResult(
                success=False,
                summary="",
                error=f"missing required field(s): {', '.join(missing)}",
            )
        date_str = str(params["date"]).strip()
        time_str = str(params["time"]).strip()
        if not _DATE_RE.match(date_str):
            return ToolResult(success=False, summary="", error="date must be YYYY-MM-DD")
        if not _TIME_RE.match(time_str):
            return ToolResult(success=False, summary="", error="time must be HH:MM")
        try:
            d = date.fromisoformat(date_str)
            hh, mm = (int(x) for x in time_str.split(":"))
            start = datetime.combine(d, datetime.min.time()).replace(hour=hh, minute=mm)
        except ValueError as e:
            return ToolResult(success=False, summary="", error=f"invalid date/time: {e}")
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return ToolResult(success=False, summary="", error="time out of range")

        event_id = uuid.uuid4().hex[:12]
        log.info(
            "booking.create_event event_id=%s business=%s call=%s when=%sT%s title=%r",
            event_id,
            business_id,
            call_id,
            date_str,
            time_str,
            params["title"],
        )
        # Future hook: integrate Composio calendar here.
        summary = f"Booked {params['title']} for {params['caller_name']} on {date_str} at {time_str}."
        return ToolResult(
            success=True,
            summary=summary,
            data={
                "event_id": event_id,
                "start": start.isoformat(),
                "title": params["title"],
                "caller_name": params["caller_name"],
            },
        )