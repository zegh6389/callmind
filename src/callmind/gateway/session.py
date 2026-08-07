from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque

import numpy as np
from fastapi import WebSocket

from ..audio.codec import (
    TELEPHONY_FRAME_BYTES,
    TELEPHONY_RATE,
    chunk_bytes,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    resample,
)
from ..audio.vad import EnergyVAD, VADEventType
from ..config import Settings
from ..llm.minimax import MinimaxChat
from ..stt.engine import WhisperSTT
from ..telephony.base import CallStart, CallStop, MediaChunk, TelephonyAdapter
from ..tts.minimax import MinimaxTTS

log = logging.getLogger("callmind.session")

STT_RATE = 16000
FRAME_SECONDS = 0.02

SYSTEM_PROMPT = (
    "You are a friendly phone support agent. Keep every reply short and "
    "conversational: one to three sentences. Never use markdown, bullet "
    "points, emojis, or stage directions. If you do not know something, "
    "say so and offer to connect the caller with a human."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, max_len: int = 180) -> tuple[list[str], str]:
    parts = _SENTENCE_SPLIT.split(text)
    if len(parts) == 1:
        if len(text) >= max_len:
            return [text], ""
        return [], text
    complete = [p.strip() + " " for p in parts[:-1] if p.strip()]
    return complete, parts[-1]


class CallSession:
    def __init__(
        self,
        ws: WebSocket,
        adapter: TelephonyAdapter,
        settings: Settings,
        stt: WhisperSTT,
        llm: MinimaxChat,
        tts: MinimaxTTS,
    ) -> None:
        self.ws = ws
        self.adapter = adapter
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts

        self.call_id = ""
        self.from_number: str | None = None
        self.state = "listening"
        self.history: deque[tuple[str, str]] = deque(maxlen=settings.memory_window)

        self._vad = EnergyVAD(
            energy_dbfs=settings.vad_energy_dbfs,
            start_frames=settings.vad_start_frames,
            end_frames=settings.vad_end_frames,
        )
        self._preroll: deque[np.ndarray] = deque(maxlen=settings.vad_preroll_frames)
        self._speech_chunks: list[np.ndarray] = []
        self._frame_leftover = b""
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._cancel = asyncio.Event()
        self._response_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._seq = 0

    async def run(self) -> None:
        self._sender_task = asyncio.create_task(self._sender_loop())
        try:
            if self.settings.greeting:
                self._start_response_text(self.settings.greeting)
            await self._receive_loop()
        finally:
            await self.close()

    async def _receive_loop(self) -> None:
        while True:
            raw = await self.ws.receive()
            if raw["type"] == "websocket.disconnect":
                break
            message = raw.get("text") if raw.get("text") is not None else raw.get("bytes")
            if message is None:
                continue
            event = self.adapter.parse(message)
            if isinstance(event, CallStart):
                self.call_id = event.call_id
                self.from_number = event.from_number
                log.info("call started id=%s from=%s", event.call_id, event.from_number)
            elif isinstance(event, MediaChunk):
                self._on_audio(event.payload)
            elif isinstance(event, CallStop):
                log.info("call stopped id=%s", self.call_id)
                break

    def _on_audio(self, mulaw_bytes: bytes) -> None:
        pcm8k = mulaw_to_pcm16(mulaw_bytes)
        data = pcm8k.tobytes()
        if self._frame_leftover:
            data = self._frame_leftover + data
        frames = chunk_bytes(data, TELEPHONY_FRAME_BYTES)
        if len(data) % TELEPHONY_FRAME_BYTES:
            self._frame_leftover = frames.pop()
        else:
            self._frame_leftover = b""

        for raw_frame in frames:
            frame = np.frombuffer(raw_frame, dtype=np.int16)
            vad_event = self._vad.process_frame(frame)

            if self._vad.in_speech:
                self._speech_chunks.append(frame)
            else:
                self._preroll.append(frame)

            if vad_event is None:
                continue
            if vad_event.type == VADEventType.SPEECH_START:
                if self.state == "speaking":
                    self._barge_in()
                self._speech_chunks = list(self._preroll)
                self._speech_chunks.append(frame)
            elif vad_event.type == VADEventType.SPEECH_END:
                self._finish_utterance()

    def _finish_utterance(self) -> None:
        if not self._speech_chunks:
            return
        utterance8k = np.concatenate(self._speech_chunks)
        self._speech_chunks = []
        duration_ms = len(utterance8k) * 1000 // TELEPHONY_RATE
        if duration_ms < self.settings.vad_min_speech_ms:
            return
        if self._response_task and not self._response_task.done():
            log.debug("utterance dropped: response already running")
            return
        utterance16k = resample(utterance8k, TELEPHONY_RATE, STT_RATE)
        self._response_task = asyncio.create_task(self._respond_audio(utterance16k))

    def _start_response_text(self, text: str) -> None:
        self._cancel.clear()
        self._response_task = asyncio.create_task(self._speak_llm_reply(text, skip_llm=True, fixed_reply=text))

    async def _respond_audio(self, pcm16k: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            text = await self.stt.transcribe(pcm16k)
        except Exception:
            log.exception("STT failed")
            return
        if not text:
            return
        log.info("STT %.0fms: %s", (time.perf_counter() - t0) * 1000, text)
        self.history.append(("user", text))
        await self._speak_llm_reply(text)

    async def _speak_llm_reply(self, user_text: str, skip_llm: bool = False, fixed_reply: str = "") -> None:
        self._cancel.clear()
        self.state = "responding"
        reply_parts: list[str] = []
        try:
            if skip_llm:
                await self._speak_text(fixed_reply)
                reply_parts.append(fixed_reply)
            else:
                pending = ""
                async for delta in self.llm.stream_chat(self._messages(user_text)):
                    if self._cancel.is_set():
                        break
                    pending += delta
                    chunks, pending = split_sentences(pending)
                    for chunk in chunks:
                        reply_parts.append(chunk)
                        await self._speak_text(chunk)
                        if self._cancel.is_set():
                            break
                    if self._cancel.is_set():
                        break
                if pending and not self._cancel.is_set():
                    reply_parts.append(pending)
                    await self._speak_text(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("response generation failed")
        finally:
            reply = "".join(reply_parts).strip()
            if reply and not self._cancel.is_set():
                self.history.append(("assistant", reply))
                log.info("agent: %s", reply)
            self.state = "listening"

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for role, content in self.history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    async def _speak_text(self, text: str) -> None:
        try:
            async for pcm_bytes in self.tts.synthesize_stream(text):
                if self._cancel.is_set():
                    return
                samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                samples8k = resample(samples, self.settings.tts_sample_rate, TELEPHONY_RATE)
                mulaw = pcm16_to_mulaw(samples8k)
                for frame in chunk_bytes(mulaw, TELEPHONY_FRAME_BYTES):
                    if self._cancel.is_set():
                        return
                    if self.state != "speaking":
                        self.state = "speaking"
                    await self._send_queue.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("TTS failed for chunk: %r", text[:80])

    def _barge_in(self) -> None:
        log.info("barge-in: caller spoke over agent")
        self._cancel.set()
        self.state = "listening"
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        clear = self.adapter.clear_message()
        if clear is not None:
            asyncio.create_task(self.ws.send_json(clear))

    async def _sender_loop(self) -> None:
        try:
            while True:
                frame = await self._send_queue.get()
                if self._cancel.is_set():
                    continue
                await self.ws.send_json(self.adapter.media_message(frame, self._seq))
                self._seq += 1
                await asyncio.sleep(FRAME_SECONDS)
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        self._cancel.set()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
        log.info("session closed id=%s", self.call_id)
