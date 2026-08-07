from __future__ import annotations

# audioop is stdlib-only until 3.13 (removed there). Pinned <3.13 in
# pyproject; if you upgrade past 3.12, replace ulaw2lin/lin2ulaw below.
import audioop
import math

import numpy as np
from scipy.signal import resample_poly

TELEPHONY_RATE = 8000
STT_RATE = 16000
FRAME_MS = 20
TELEPHONY_FRAME_BYTES = TELEPHONY_RATE * FRAME_MS // 1000 * 2


def mulaw_to_pcm16(mulaw_bytes: bytes) -> np.ndarray:
    pcm = audioop.ulaw2lin(mulaw_bytes, 2)
    return np.frombuffer(pcm, dtype=np.int16)


def pcm16_to_mulaw(samples: np.ndarray) -> bytes:
    data = np.ascontiguousarray(samples, dtype=np.int16).tobytes()
    return audioop.lin2ulaw(data, 2)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    g = math.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    out = resample_poly(samples.astype(np.float64), up, down)
    return np.clip(out, -32768, 32767).astype(np.int16)


def pcm16_to_float32(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def chunk_bytes(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -float("inf")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 0:
        return -float("inf")
    return 20.0 * math.log10(rms / 32768.0)
