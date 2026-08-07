from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..audio.codec import pcm16_to_float32

log = logging.getLogger("callmind.stt")


class WhisperSTT:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device, compute = self._resolve_runtime()
        log.info("loading faster-whisper %s device=%s compute=%s", self.model_size, device, compute)
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute)

    def _resolve_runtime(self) -> tuple[str, str]:
        device = self.device
        compute = self.compute_type
        if device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                else:
                    device = "cpu"
            except (ImportError, OSError, RuntimeError, ValueError):
                device = "cpu"
        if compute == "auto":
            compute = "int8_float16" if device == "cuda" else "int8"
        return device, compute

    def transcribe_sync(self, pcm16k: np.ndarray) -> str:
        self.load()
        audio = pcm16_to_float32(pcm16k)
        language = self.language or None
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, pcm16k: np.ndarray) -> str:
        async with self._lock:
            return await asyncio.to_thread(self.transcribe_sync, pcm16k)

    def close(self) -> None:
        """Release the underlying model so shutdown is clean."""
        self._model = None
