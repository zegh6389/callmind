from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket

from ..admin.router import router as admin_router
from ..admin.store import BusinessStore
from ..brain.memory import MemoryStore
from ..config import get_settings
from ..llm.embeddings import MinimaxEmbeddings
from ..llm.minimax import MinimaxChat
from ..stt.engine import WhisperSTT
from ..tools.router import ToolRouter
from ..telephony import create_adapter
from ..telephony.client import TelnyxAPI
from ..tts.minimax import MinimaxTTS
from .session import CallSession

log = logging.getLogger("callmind.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stt = WhisperSTT(
        model_size=settings.stt_model_size,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=settings.stt_language,
    )
    await asyncio.to_thread(stt.load)

    llm = MinimaxChat(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    tts = MinimaxTTS(
        api_key=settings.tts_api_key or settings.llm_api_key,
        base_url=settings.tts_base_url,
        endpoint=settings.tts_endpoint,
        model=settings.tts_model,
        voice_id=settings.tts_voice_id,
        sample_rate=settings.tts_sample_rate,
        speed=settings.tts_speed,
        volume=settings.tts_volume,
        pitch=settings.tts_pitch,
    )
    embeddings = MinimaxEmbeddings(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        endpoint=settings.embedding_endpoint,
        model=settings.embedding_model,
        embedding_type=settings.embedding_type,
    )

    app.state.settings = settings
    app.state.stt = stt
    app.state.llm = llm
    app.state.tts = tts
    app.state.embeddings = embeddings
    app.state.tool_router = ToolRouter()
    app.state.telnyx = TelnyxAPI(
        api_key=settings.telnyx_api_key,
        base_url=settings.telnyx_api_base,
    )
    app.state.business_store = BusinessStore(settings.memory_db_path)
    app.state.memory = MemoryStore(settings.memory_db_path)
    log.info("gateway ready (provider=%s)", settings.telephony_provider)
    yield
    await llm.close()
    await tts.close()
    await embeddings.close()
    await app.state.telnyx.close()


app = FastAPI(title="CallMind Voice Gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(admin_router)


@app.post("/telnyx/webhook")
async def telnyx_webhook(request: Request) -> dict:
    body = await request.json()
    data = body.get("data", {})
    event_type = data.get("event_type")
    payload = data.get("payload", {})
    log.info("telnyx webhook: %s", event_type)

    if event_type == "call.initiated":
        call_control_id = payload.get("call_control_id")
        settings = app.state.settings
        if call_control_id and settings.public_ws_url:
            asyncio.create_task(
                app.state.telnyx.answer_with_stream(call_control_id, settings.public_ws_url)
            )
        else:
            log.error("cannot answer call: missing call_control_id or public_ws_url")
    return {"status": "ok"}


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket) -> None:
    await ws.accept()
    st = app.state
    session = CallSession(
        ws=ws,
        adapter=create_adapter(st.settings),
        settings=st.settings,
        stt=st.stt,
        llm=st.llm,
        tts=st.tts,
        embeddings=st.embeddings,
        tool_router=st.tool_router,
    )
    await session.run()
