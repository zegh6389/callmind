"""Mock telephony provider: drives /ws/call without Telnyx.

Sends Twilio-style start/media/stop events with silent audio, records
whatever the gateway speaks back into a WAV file.

Usage:
    uv run python scripts/mock_call.py --url ws://localhost:8000/ws/call --seconds 10
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import wave

import websockets

SILENCE_FRAME = base64.b64encode(b"\xff" * 160).decode()


async def run(url: str, seconds: float, out_wav: str) -> None:
    frames_to_send = int(seconds * 50)
    received_pcm = bytearray()

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "sequence_number": "1",
                    "start": {
                        "call_control_id": "v2:MOCK-CALL",
                        "call_session_id": "MOCK-SESSION",
                        "from": "+15550000001",
                        "to": "+15550000002",
                        "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
                    },
                    "stream_id": "MOCK-STREAM",
                }
            )
        )

        async def receiver() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("event") != "media":
                    continue
                payload = msg.get("media", {}).get("payload", "")
                if payload:
                    received_pcm.extend(audioop.ulaw2lin(base64.b64decode(payload), 2))

        recv_task = asyncio.create_task(receiver())

        for i in range(frames_to_send):
            await ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "sequence_number": str(i + 2),
                        "media": {"track": "inbound", "chunk": str(i), "payload": SILENCE_FRAME},
                        "stream_id": "MOCK-STREAM",
                    }
                )
            )
            await asyncio.sleep(0.02)

        await ws.send(
            json.dumps(
                {
                    "event": "stop",
                    "stop": {"call_control_id": "v2:MOCK-CALL"},
                    "stream_id": "MOCK-STREAM",
                }
            )
        )
        await asyncio.sleep(1.0)
        recv_task.cancel()

    if received_pcm:
        with wave.open(out_wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8000)
            f.writeframes(bytes(received_pcm))
        print(f"saved {len(received_pcm) / 2 / 8000:.1f}s of agent audio -> {out_wav}")
    else:
        print("no agent audio received (check MiniMax keys / greeting config)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock telephony provider for CallMind gateway")
    parser.add_argument("--url", default="ws://localhost:8000/ws/call")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", default="mock_call_out.wav")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.seconds, args.out))


if __name__ == "__main__":
    main()
