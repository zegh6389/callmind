from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger("callmind.llm")


class MinimaxChat:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        endpoint: str = "/text/chatcompletion_v2",
        model: str = "MiniMax-Text-01",
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        async with self._client.stream("POST", self.endpoint, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"MiniMax LLM HTTP {resp.status_code}: {body[:500]!r}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    log.warning("unparseable LLM chunk: %s", data[:200])
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta

    async def close(self) -> None:
        await self._client.aclose()
