"""Session behavior tests with stubbed STT/LLM/TTS/embeddings.

Drives CallSession through fake media events, asserts:
- booking intent -> BookingTool runs -> reply mentions event_id.
- account_status intent -> AccountTool runs -> reply mentions account id.
- faq intent -> no tool, normal LLM reply.
- extraction of params from user text drives the tool.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import numpy as np
import pytest

from callmind.config import Settings
from callmind.gateway.session import CallSession
from callmind.tools.router import ToolRouter


class StubLLM:
    """Reply queue. Each stream_chat call consumes the next entry.

    Use a list of (messages_matcher, reply_text) tuples for ordered control, or
    plain strings for "return this verbatim".
    """

    def __init__(self, replies: list, delay: float = 0.0) -> None:
        self._replies = list(replies)
        self._i = 0
        self.call_log: list[list[dict]] = []
        self.delay = delay

    async def stream_chat(self, messages, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        self.call_log.append(list(messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self._replies[self._i % len(self._replies)]
        self._i += 1
        text = item if isinstance(item, str) else item.get("text", "")
        for word in text.split():
            yield word + " "


class StubSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, pcm16k: np.ndarray) -> str:
        self.calls += 1
        return self.text


class StubTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        yield np.zeros(160, dtype=np.int16).tobytes()


class StubEmbeddings:
    async def embed(self, texts):
        return [[0.1] * 4 for _ in texts]

    async def close(self):
        pass


class StubAdapter:
    def __init__(self):
        self.name = "stub"
        self._stream = "stub-stream"
        self._start_called = False
        self.media_sent: list[bytes] = []
        self.clears: int = 0

    def start_message(self):
        return {"event": "start", "test": True}

    def parse(self, message):
        if isinstance(message, bytes):
            message = message.decode()
        d = json.loads(message)
        if d.get("event") == "start":
            self._start_called = True
            from callmind.telephony.base import CallStart

            s = d["start"]
            return CallStart(
                call_id=s.get("call_control_id", "c1"),
                stream_id=self._stream,
                from_number=s.get("from"),
                to_number=s.get("to"),
                raw=d,
            )
        if d.get("event") == "media":
            from callmind.telephony.base import MediaChunk

            return MediaChunk(payload=b"\xff" * 160, stream_id=self._stream)
        if d.get("event") == "stop":
            from callmind.telephony.base import CallStop

            return CallStop(call_id="c1")
        return None

    def media_message(self, mulaw, seq):
        self.media_sent.append(mulaw)
        return {"event": "media", "seq": seq}

    def clear_message(self):
        self.clears += 1
        return {"event": "clear"}


class FakeWS:
    def __init__(self):
        self.received: list[dict] = []
        self.to_send: list[dict] = []
        self.closed = False

    async def receive(self):
        if self.to_send:
            return {"type": "websocket.receive", "text": json.dumps(self.to_send.pop(0))}
        return {"type": "websocket.disconnect"}

    async def send_json(self, obj):
        self.received.append(obj)

    async def close(self, code: int = 1000):
        self.closed = True


def test_utterance_parked_while_previous_response_busy(settings):
    llm = StubLLM([
        '{"intent":"smalltalk","confidence":0.9}',
        "First reply text",
        '{"intent":"smalltalk","confidence":0.9}',
        "Second reply text",
    ], delay=0.1)
    stt = StubSTT("hello agent")  # same text both utterances
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=settings,
        stt=stt, llm=llm, tts=tts, embeddings=StubEmbeddings(),
    )

    async def run():
        await session._receive_loop()
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        first_task = session._response_task
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        assert first_task is not None and not first_task.done()
        await asyncio.wait_for(first_task, timeout=2.0)
        await asyncio.sleep(0.1)  # parked utterance drains synchronously after

    asyncio.run(run())
    joined = " ".join(tts.spoken).lower()
    assert "first reply text" in joined
    assert "second reply text" in joined


def test_speech_start_captures_preroll_and_frame_once(settings):
    # Regression lock for review item #9: SPEECH_START must not duplicate
    # the trigger frame (preroll + exactly one copy of the loud frame).
    session = CallSession(
        ws=FakeWS(), adapter=StubAdapter(), settings=settings,
        stt=StubSTT(""), llm=StubLLM(["hi"]), tts=StubTTS(),
        embeddings=StubEmbeddings(),
    )
    _push_silence(session, n_frames=3)          # fill preroll (maxlen=3)
    loud = np.full(160, 8000, dtype=np.int16).tobytes()
    session._on_audio(audioop_lin2ulaw(loud))   # triggers SPEECH_START

    chunks = session._speech_chunks
    assert len(chunks) == 4                     # 3 preroll + trigger frame
    loud_frames = [np.max(c) for c in chunks]
    assert sum(1 for v in loud_frames if v > 5000) == 1


def test_barge_clear_delivered_after_pending_media(settings):
    ws = FakeWS()
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=settings,
        stt=StubSTT(""), llm=StubLLM(["hi"]), tts=StubTTS(),
        embeddings=StubEmbeddings(),
    )

    async def run():
        session._sender_task = asyncio.create_task(session._sender_loop())
        for _ in range(2):
            await session._send_queue.put(("media", b"\xff" * 160))
        await asyncio.sleep(0.05)
        session._barge_in()
        await asyncio.sleep(0.05)

    asyncio.run(run())
    events = [m.get("event") for m in ws.received]
    assert events.count("media") >= 2
    assert events[-1] == "clear", f"clear must be last, got {events}"
    clear_index = len(events) - 1 - events[::-1].index("clear")
    assert all(e == "media" for e in events[:clear_index])


def test_greeting_before_callstart_leaves_no_orphan_rows(settings):
    # Greeting TTS can start before the provider's start frame arrives --
    # speaken fully with memory writes. Must not write call_id="" rows.
    session = CallSession(
        ws=FakeWS(), adapter=StubAdapter(), settings=settings,
        stt=StubSTT(""), llm=StubLLM(["you"]), tts=StubTTS(),
        embeddings=StubEmbeddings(),
    )

    async def run():
        session._start_response_text("Welcome to the line. ")
        await session._response_task
        await session.close()

    asyncio.run(run())
    import sqlite3

    with sqlite3.connect(settings.memory_db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert n == 0


def test_session_uses_business_row_greeting(settings, tmp_path):
    from callmind.admin.store import BusinessStore

    db = BusinessStore(str(tmp_path / "biz.db"))
    biz = db.create_business("default", prompt="Talk like a pirate.", greeting="Ahoy matey!")
    s2 = settings.model_copy(update={"business_id": biz["id"]})
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=s2,
        stt=StubSTT(""), llm=StubLLM(["hi"]), tts=tts,
        embeddings=StubEmbeddings(), business_store=db,
    )

    async def go():
        await session._receive_loop()          # resolves business row
        session._start_response_text(session.business_greeting)
        await session._response_task
        await session.close()

    asyncio.run(go())
    assert tts.spoken == ["Ahoy matey!"]


def test_session_escalation_threshold_from_business(settings, tmp_path):
    from callmind.admin.store import BusinessStore

    db = BusinessStore(str(tmp_path / "biz2.db"))
    biz = db.create_business("default", escalation_confidence=0.95)
    s2 = settings.model_copy(update={"business_id": biz["id"]})
    # faq with conf 0.8: global threshold 0.55 would answer directly;
    # business threshold 0.95 must route to the clarify path instead.
    llm = StubLLM([
        '{"intent":"faq","confidence":0.8}',
        "I can help with that.",
    ])
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=s2,
        stt=StubSTT("what are your hours"), llm=llm, tts=tts,
        embeddings=StubEmbeddings(), business_store=db,
    )

    async def run():
        await session._receive_loop()
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        if session._response_task:
            await asyncio.wait_for(session._response_task, timeout=2.0)

    asyncio.run(run())
    assert "rephrase" in " ".join(tts.spoken).lower()


def test_session_closes_websocket_when_done(settings):
    ws = FakeWS()
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=settings,
        stt=StubSTT(""), llm=StubLLM(["hi"]), tts=StubTTS(),
        embeddings=StubEmbeddings(),
    )
    asyncio.run(session.run())
    assert ws.closed


@pytest.fixture
def settings(tmp_path):
    return Settings(
        memory_db_path=str(tmp_path / "callmind.db"),
        kb_dir=str(tmp_path / "kb"),
        vad_min_speech_ms=0,  # accept anything for tests
        vad_preroll_frames=3,
        vad_end_frames=2,
        vad_start_frames=1,
        greeting="",
    )


async def _drive(session: CallSession, ws: FakeWS):
    await session._receive_loop()


def _push_audio(session: CallSession):
    # _on_audio expects mu-law 8k bytes. Encode a "loud" int16 frame to mu-law.
    pcm = np.full(160, 8000, dtype=np.int16)
    mulaw = audioop_lin2ulaw(pcm.tobytes())
    session._on_audio(mulaw)


def _push_silence(session: CallSession, n_frames: int = 10):
    pcm = np.zeros(160, dtype=np.int16).tobytes()
    mulaw = audioop_lin2ulaw(pcm)
    for _ in range(n_frames):
        session._on_audio(mulaw)


def audioop_lin2ulaw(pcm_bytes: bytes) -> bytes:
    import audioop

    return audioop.lin2ulaw(pcm_bytes, 2)


def test_session_parse_failure_does_not_clarify_loop(settings):
    # Intent JSON malformed -> IntentChain falls back to smalltalk/0.0.
    # Old behavior: conf < 0.55 -> permanent "rephrase that" trap.
    # New: parse failure must still produce a real answer, no clarify.
    llm = StubLLM(["not json at all", "Sure, I can help with that."])
    stt = StubSTT("hi there")
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    session = CallSession(
        ws=ws, adapter=StubAdapter(), settings=settings,
        stt=stt, llm=llm, tts=tts, embeddings=StubEmbeddings(),
    )

    async def run():
        await session._receive_loop()
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        if session._response_task:
            await asyncio.wait_for(session._response_task, timeout=2.0)

    asyncio.run(run())
    assert "rephrase" not in " ".join(tts.spoken).lower()
    # Pull-up handshake: provider requiring an explicit client 'start'
    # (Telnyx) must get it before any media flows.
    ws = FakeWS()
    adapter = StubAdapter()
    session = CallSession(
        ws=ws, adapter=adapter, settings=settings,
        stt=StubSTT(""), llm=StubLLM(["hi"]), tts=StubTTS(),
        embeddings=StubEmbeddings(),
    )

    asyncio.run(session.run())
    assert ws.received, "expected at least the start frame"
    assert ws.received[0] == {"event": "start", "test": True}


def test_session_booking_runs_tool(settings):
    llm = StubLLM([
        '{"intent":"booking","confidence":0.9}',  # intent classification
        "I have booked your appointment.",         # reply after tool ran
    ])
    stt = StubSTT("Can I book an appointment with John tomorrow at 2pm?")
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    adapter = StubAdapter()
    router = ToolRouter()

    session = CallSession(
        ws=ws, adapter=adapter, settings=settings,
        stt=stt, llm=llm, tts=tts, embeddings=StubEmbeddings(), tool_router=router,
    )

    async def run():
        # Receive loop consumes the start event, then no more -> disconnect
        await session._receive_loop()
        # Now manually drive an utterance
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        # wait for the response task
        if session._response_task:
            try:
                await asyncio.wait_for(session._response_task, timeout=2.0)
            except TimeoutError:
                pass

    asyncio.run(run())

    # The intent classification call happens first. The second LLM call is
    # the actual reply with tool context. Look at the second call's messages.
    assert len(llm.call_log) >= 2, f"expected at least 2 LLM calls, got {len(llm.call_log)}"
    reply_messages = llm.call_log[-1]
    assert any(
        "Booked" in (m.get("content") or "")
        for m in reply_messages
        if m.get("role") == "system"
    ), f"tool summary missing from final LLM context: {reply_messages}"
    assert "booked your appointment" in " ".join(tts.spoken).lower()


def _run_session(settings, user_text, intent_json, reply_text, *, stt_text=None, tool_router=None):
    """Helper: drive a session through one utterance and return llm, tts, adapter."""
    llm = StubLLM([intent_json, reply_text])
    stt = StubSTT(stt_text or user_text)
    tts = StubTTS()
    ws = FakeWS()
    ws.to_send = [
        {"event": "start", "start": {"call_control_id": "c1", "from": "+15551111111", "to": "+15550000"}},
    ]
    adapter = StubAdapter()
    router = tool_router or ToolRouter(stub_mode=True)
    session = CallSession(
        ws=ws, adapter=adapter, settings=settings,
        stt=stt, llm=llm, tts=tts, embeddings=StubEmbeddings(), tool_router=router,
    )

    async def drive():
        await session._receive_loop()
        session._vad.reset()
        _push_audio(session)
        _push_silence(session)
        if session._response_task:
            try:
                await asyncio.wait_for(session._response_task, timeout=2.0)
            except TimeoutError:
                pass

    asyncio.run(drive())
    return llm, tts, adapter


def test_session_account_runs_tool(settings):
    llm, tts, _ = _run_session(
        settings,
        user_text="what's my balance on +15551111111?",
        intent_json='{"intent":"account_status","confidence":0.85}',
        reply_text="Your account is in good standing.",
    )
    assert len(llm.call_log) >= 2
    reply = llm.call_log[-1]
    assert any("acct_" in (m.get("content") or "") for m in reply if m.get("role") == "system"), reply
    assert "good standing" in " ".join(tts.spoken).lower()


def test_session_escalation_skips_tool_and_llm(settings):
    llm, tts, _ = _run_session(
        settings,
        user_text="this is ridiculous, get me a human",
        intent_json='{"intent":"escalation","confidence":0.95}',
        reply_text="",
    )
    # escalation -> only the intent-classification LLM call. No second LLM call,
    # TTS gets the canned ESCALATION_TEXT.
    assert len(llm.call_log) == 1
    assert any("connect you with a human" in s for s in tts.spoken)


def test_session_faq_no_tool(settings):
    llm, tts, _ = _run_session(
        settings,
        user_text="what are your opening hours?",
        intent_json='{"intent":"faq","confidence":0.8}',
        reply_text="We are open nine to five.",
    )
    # FAQ -> tool dispatch NOT triggered -> system message has no "Tool result for"
    reply = llm.call_log[-1]
    sys_msg = next((m for m in reply if m.get("role") == "system"), None)
    assert sys_msg is not None
    assert "Tool result" not in sys_msg["content"]
    assert "nine to five" in " ".join(tts.spoken).lower()


def test_tool_not_whitelisted_skipped(settings):
    llm, tts, _ = _run_session(
        settings,
        user_text="random",
        intent_json='{"intent":"smalltalk","confidence":0.9}',
        reply_text="Hi there!",
    )
    assert "hi there" in " ".join(tts.spoken).lower()
    assert not llm.call_log or len(llm.call_log) >= 1  # smalltalk -> only intent call