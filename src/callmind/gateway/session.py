from __future__ import annotations

import asyncio
import logging
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
from ..brain import IntentChain, MemoryStore, VectorStore
from ..config import Settings
from ..llm.embeddings import MinimaxEmbeddings
from ..llm.minimax import MinimaxChat
from ..stt.engine import WhisperSTT
from ..telephony.base import CallStart, CallStop, MediaChunk, TelephonyAdapter
from ..tools.router import ToolRouter
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

ESCALATION_TEXT = (
    "I'll connect you with a human representative right away. "
    "Please hold for a moment."
)

_SENTENCE_SPLIT = re = __import__("re").compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, max_len: int = 180) -> tuple[list[str], str]:
    parts = re.split(text)
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
        embeddings: MinimaxEmbeddings,
        tool_router: ToolRouter | None = None,
    ) -> None:
        self.ws = ws
        self.adapter = adapter
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.embeddings = embeddings

        self.call_id = ""
        self.from_number: str | None = None
        self.state = "listening"

        self.intent_chain = IntentChain(llm)
        self.memory = MemoryStore(settings.memory_db_path)
        self.kb = VectorStore(settings.business_id, settings.kb_dir)
        self.tool_router = tool_router or ToolRouter()

        self.history: deque[tuple[str, str]] = deque(maxlen=settings.memory_window)
        self._loaded_history = False

        self._vad = EnergyVAD(
            energy_dbfs=settings.vad_energy_dbfs,
            start_frames=settings.vad_start_frames,
            end_frames=settings.vad_end_frames,
        )
        self._preroll: deque[np.ndarray] = deque(maxlen=settings.vad_preroll_frames)
        self._speech_chunks: list[np.ndarray] = []
        self._parked_utterance: np.ndarray | None = None
        self._frame_leftover = b""
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._cancel = asyncio.Event()
        self._response_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._seq = 0

    async def run(self) -> None:
        self._sender_task = asyncio.create_task(self._sender_loop())
        if start := self.adapter.start_message():
            await self.ws.send_json(start)
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
                self.call_id = event.call_id or self.call_id
                self.from_number = event.from_number
                self.memory.start_conversation(self.call_id, self.settings.business_id, self.from_number)
                self._seed_history()
                log.info("call started id=%s from=%s", self.call_id, self.from_number)
            elif isinstance(event, MediaChunk):
                self._on_audio(event.payload)
            elif isinstance(event, CallStop):
                log.info("call stopped id=%s", self.call_id)
                break

    def _seed_history(self) -> None:
        if self._loaded_history:
            return
        prior = self.memory.load_recent(
            self.settings.business_id, self.from_number, limit=self.settings.memory_window
        )
        for role, content in prior:
            self.history.append((role, content))
        self._loaded_history = True

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
        utterance16k = resample(utterance8k, TELEPHONY_RATE, STT_RATE)
        if self._response_task and not self._response_task.done():
            log.debug("utterance parked: response already running")
            self._parked_utterance = utterance16k
            return
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
        self.memory.append_message(self.call_id, "user", text)
        await self._handle_turn(text)

    async def _handle_turn(self, user_text: str) -> None:
        try:
            intent = await self.intent_chain.classify(user_text, list(self.history))
        except Exception:
            log.exception("intent classification failed")
            intent = None

        if intent and intent.label == "escalation":
            log.info("intent=escalation -> human handoff")
            await self._speak_llm_reply(user_text, skip_llm=True, fixed_reply=ESCALATION_TEXT)
            return
        if intent and intent.confidence < self.settings.escalation_confidence_threshold:
            log.info("low confidence %.2f -> ask to clarify", intent.confidence)
            await self._speak_llm_reply(
                user_text,
                skip_llm=True,
                fixed_reply="Sorry, I want to make sure I help with the right thing. Could you rephrase that?",
            )
            return

        rag_context: str | None = None
        if intent and intent.label in ("booking", "account_status"):
            params = self.tool_router.extract_params(intent.label, user_text)
            result = await self.tool_router.dispatch(
                intent.label,
                params,
                call_id=self.call_id,
                business_id=self.settings.business_id,
            )
            if result and result.success:
                log.info("tool %s ok: %s", intent.label, result.summary)
                rag_context = f"[Tool result for {intent.label}]\n{result.summary}"
                self.memory.append_message(self.call_id, "tool", f"{intent.label}: {result.summary}")
            elif result and not result.success:
                log.warning("tool %s failed: %s", intent.label, result.error)
                rag_context = f"[Tool {intent.label} failed: {result.error}]"
        if intent and intent.label == "faq" and not self.kb.is_empty():
            rag_context = await self._retrieve_context(user_text)
        await self._speak_llm_reply(user_text, rag_context=rag_context)

    async def _retrieve_context(self, query: str) -> str | None:
        try:
            vecs = await self.embeddings.embed([query])
        except Exception:
            log.exception("embedding query failed")
            return None
        if not vecs:
            return None
        hits = self.kb.search(vecs[0], top_k=self.settings.retrieval_top_k)
        good = [h for h in hits if h[1] >= self.settings.retrieval_min_score]
        if not good:
            return None
        return "\n\n".join(f"[source: {src}]\n{text}" for text, _score, src in good)

    async def _speak_llm_reply(
        self,
        user_text: str,
        skip_llm: bool = False,
        fixed_reply: str = "",
        rag_context: str | None = None,
    ) -> None:
        self._cancel.clear()
        self.state = "responding"
        reply_parts: list[str] = []
        try:
            if skip_llm:
                await self._speak_text(fixed_reply)
                reply_parts.append(fixed_reply)
            else:
                pending = ""
                messages = self._messages(user_text, rag_context)
                async for delta in self.llm.stream_chat(messages):
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
                self.memory.append_message(self.call_id, "assistant", reply)
                log.info("agent: %s", reply)
            self.state = "listening"
            if not self._cancel.is_set():
                parked = self._parked_utterance
                self._parked_utterance = None
                if parked is not None and self._response_task is asyncio.current_task():
                    await self._respond_audio(parked)

    def _messages(self, user_text: str, rag_context: str | None) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT
        if rag_context:
            system += "\n\nUse the following knowledge base context to answer. Do not invent facts:\n" + rag_context
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
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
        self._parked_utterance = None
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
        if self.call_id:
            self.memory.end_conversation(self.call_id)
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
        try:
            await self.ws.close()
        except Exception:
            log.debug("ws already closed", exc_info=True)
        log.info("session closed id=%s", self.call_id)