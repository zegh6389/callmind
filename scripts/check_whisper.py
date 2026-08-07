import time

import numpy as np

t0 = time.perf_counter()
from faster_whisper import WhisperModel

device = "cpu"
try:
    import ctranslate2

    n = ctranslate2.get_cuda_device_count()
    print("cuda devices:", n)
    if n > 0:
        device = "cuda"
except Exception as e:  # noqa: BLE001
    print("cuda probe failed:", e)

compute = "int8_float16" if device == "cuda" else "int8"
print(f"loading faster-whisper small on {device} ({compute})...")
t1 = time.perf_counter()
model = WhisperModel("small", device=device, compute_type=compute)
print(f"model load: {time.perf_counter() - t1:.1f}s")

sr = 16000
t = np.arange(sr * 3) / sr
tone = (np.sin(2 * np.pi * 300 * t) * 0.3 * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
t2 = time.perf_counter()
segments, info = model.transcribe(tone, beam_size=1)
text = " ".join(s.text for s in segments)
dt = time.perf_counter() - t2
print(f"transcribe #1 (warmup): {dt * 1000:.0f}ms text={text!r} lang={info.language}")
t3 = time.perf_counter()
segments, info = model.transcribe(tone, beam_size=1)
text = " ".join(s.text for s in segments)
dt2 = time.perf_counter() - t3
print(f"transcribe #2 (steady): {dt2 * 1000:.0f}ms text={text!r}")
print(f"total: {time.perf_counter() - t0:.1f}s")
