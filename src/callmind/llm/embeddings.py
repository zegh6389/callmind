from __future__ import annotations

import logging

import httpx

log = logging.getLogger("callmind.embeddings")


class MinimaxEmbeddings:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        endpoint: str = "/v1/embeddings",
        model: str = "embo-01",
        embedding_type: str = "db",
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.embedding_type = embedding_type
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "texts": texts,
            "type": self.embedding_type,
        }
        resp = await self._client.post(self.endpoint, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"embeddings HTTP {resp.status_code}: {resp.text[:500]}")
        obj = resp.json()
        base = obj.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            raise RuntimeError(f"embeddings error: {base}")
        vectors = obj.get("vectors") or []
        if not vectors:
            raise RuntimeError("embeddings returned no vectors")
        return vectors

    async def close(self) -> None:
        await self._client.aclose()