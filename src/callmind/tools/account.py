from __future__ import annotations

import hashlib
import logging

from .base import Tool, ToolResult

log = logging.getLogger("callmind.tools.account")


class AccountTool(Tool):
    name = "account.get_status"
    description = "Look up account status for the caller."

    async def run(
        self,
        params: dict,
        *,
        call_id: str,
        business_id: str,
    ) -> ToolResult:
        phone = str(params.get("caller_phone") or params.get("account_id") or "").strip()
        if not phone:
            return ToolResult(success=False, summary="", error="missing caller_phone or account_id")

        # Deterministic stub: hash phone -> canned state. Real impl: call CRM.
        digest = hashlib.sha256(phone.encode("utf-8")).digest()
        bucket = digest[0] % 3
        status = ("active", "payment_due", "closed")[bucket]
        balance = round((digest[1] / 255.0) * 499 + 1, 2)
        account_id = "acct_" + hashlib.sha1(phone.encode("utf-8")).hexdigest()[:10]

        log.info(
            "account.get_status account=%s business=%s call=%s status=%s",
            account_id,
            business_id,
            call_id,
            status,
        )
        summary = (
            f"Account {account_id} is {status}. "
            + (f"Outstanding balance: ${balance}." if status == "payment_due" else "")
        ).strip()
        return ToolResult(
            success=True,
            summary=summary,
            data={"account_id": account_id, "status": status, "balance": balance},
        )