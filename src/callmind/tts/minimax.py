from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger("callmind.tts")


class MinimaxTTS:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1",
        endpoint: str = "/t2a_v2",
        model: str = "speech-2.8-flash",
        voice_id: str = "",
        sample_rate: int = 24000,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: int = 0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.speed = speed
        self.volume = volume
        self.pitch = pitch
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def _payload(self, text: str) -> dict:
        return {
            "model": self.model,
            "text": text,
            "stream": True,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": self.volume,
                "pitch": self.pitch,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1,
            },
        }

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        async with self._client.stream("POST", self.endpoint, json=self._payload(text)) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"MiniMax TTS HTTP {resp.status_code}: {body[:500]!r}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("unparseable TTS chunk: %s", line[:200])
                    continue
                base_resp = obj.get("base_resp")
                if base_resp and base_resp.get("status_code", 0) != 0:
                    raise RuntimeError(f"MiniMax TTS error: {base_resp}")
                audio_hex = (obj.get("data") or {}).get("audio")
                if audio_hex:
                    yield bytes.fromhex(audio_hex)

    async def close(self) -> None:
        await self._client.aclose()
