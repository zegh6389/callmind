from __future__ import annotations

import logging

import httpx

log = logging.getLogger("callmind.telephony.client")


class TelnyxAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.telnyx.com") -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def answer_with_stream(
        self,
        call_control_id: str,
        stream_url: str,
        track: str = "inbound_track",
        codec: str = "PCMU",
    ) -> None:
        resp = await self._client.post(
            f"/v2/calls/{call_control_id}/actions/answer",
            json={
                "stream_url": stream_url,
                "stream_track": track,
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": codec,
            },
        )
        if resp.status_code >= 300:
            log.error("answer_with_stream failed %s: %s", resp.status_code, resp.text[:500])
        else:
            log.info("answer+stream requested for %s", call_control_id)

    async def close(self) -> None:
        await self._client.aclose()
