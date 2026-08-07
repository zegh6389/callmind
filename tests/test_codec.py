import numpy as np

from callmind.audio.codec import (
    chunk_bytes,
    dbfs,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    resample,
)


def test_ulaw_silence_code_decodes_to_zero():
    pcm = mulaw_to_pcm16(b"\xff")
    assert pcm[0] == 0


def test_ulaw_roundtrip_error_bounded():
    t = np.arange(1600) / 16000.0
    x = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    out = mulaw_to_pcm16(pcm16_to_mulaw(x))
    err = np.abs(out.astype(np.int32) - x.astype(np.int32))
    assert err.max() < 600
    assert err.mean() < 120


def test_ulaw_preserves_length():
    x = np.zeros(160, dtype=np.int16)
    assert len(pcm16_to_mulaw(x)) == 160
    assert len(mulaw_to_pcm16(pcm16_to_mulaw(x))) == 160


def test_resample_up_8k_to_16k():
    x = (np.random.default_rng(0).standard_normal(800) * 1000).astype(np.int16)
    y = resample(x, 8000, 16000)
    assert len(y) == 1600


def test_resample_down_24k_to_8k():
    x = (np.random.default_rng(1).standard_normal(2400) * 1000).astype(np.int16)
    y = resample(x, 24000, 8000)
    assert len(y) == 800


def test_resample_same_rate_is_noop():
    x = np.arange(100, dtype=np.int16)
    y = resample(x, 8000, 8000)
    assert np.array_equal(x, y)


def test_chunk_bytes():
    data = b"\x00" * 500
    chunks = chunk_bytes(data, 160)
    assert len(chunks) == 4
    assert len(chunks[-1]) == 20


def test_dbfs_full_scale_near_zero():
    x = np.full(160, 32767, dtype=np.int16)
    assert abs(dbfs(x)) < 0.01


def test_dbfs_silence_is_minus_inf():
    assert dbfs(np.zeros(160, dtype=np.int16)) == -float("inf")
