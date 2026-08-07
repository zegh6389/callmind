from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..brain import VectorStore, chunk_text
from .store import BusinessStore

log = logging.getLogger("callmind.admin.router")

router = APIRouter(prefix="/admin", tags=["admin"])


def _biz(request: Request) -> BusinessStore:
    return request.app.state.business_store


def _kb_base(request: Request) -> str:
    return request.app.state.settings.kb_dir


# --- schemas ---


class BusinessCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = ""
    greeting: str = ""


class BusinessUpdate(BaseModel):
    name: str | None = None
    prompt: str | None = None
    greeting: str | None = None
    escalation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class KbDocCreate(BaseModel):
    source: str = ""
    text: str = Field(min_length=1)


# --- businesses ---


@router.get("/businesses")
def list_businesses(request: Request) -> dict:
    return {"businesses": _biz(request).list_businesses()}


@router.post("/businesses", status_code=201)
def create_business(request: Request, body: BusinessCreate) -> dict:
    b = _biz(request).create_business(body.name, body.prompt, body.greeting)
    return {"business": b}


@router.get("/businesses/{business_id}")
def get_business(request: Request, business_id: str) -> dict:
    b = _biz(request).get_business(business_id)
    if not b:
        raise HTTPException(404, "business not found")
    return {"business": b}


@router.patch("/businesses/{business_id}")
def update_business(request: Request, business_id: str, body: BusinessUpdate) -> dict:
    sets = body.model_dump(exclude_unset=True)
    b = _biz(request).update_business(business_id, **sets)
    if not b:
        raise HTTPException(404, "business not found")
    return {"business": b}


@router.delete("/businesses/{business_id}", status_code=204)
def delete_business(request: Request, business_id: str) -> None:
    if not _biz(request).delete_business(business_id):
        raise HTTPException(404, "business not found")
    # Also drop on-disk KB for that business.
    import shutil
    from pathlib import Path

    p = Path(_kb_base(request)) / business_id
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


# --- KB ---


@router.get("/businesses/{business_id}/kb/docs")
def list_kb_docs(request: Request, business_id: str) -> dict:
    if not _biz(request).get_business(business_id):
        raise HTTPException(404, "business not found")
    return {"docs": _biz(request).list_docs(business_id)}


@router.post("/businesses/{business_id}/kb/docs", status_code=201)
async def create_kb_doc(request: Request, business_id: str, body: KbDocCreate) -> dict:
    bs = _biz(request)
    if not bs.get_business(business_id):
        raise HTTPException(404, "business not found")

    chunks = chunk_text(body.text)
    if not chunks:
        raise HTTPException(400, "document produced no chunks")

    embeddings = request.app.state.embeddings
    try:
        vectors = await embeddings.embed(chunks)
    except Exception as e:
        raise HTTPException(502, f"embedding failed: {e}") from e

    doc = bs.create_doc(business_id, body.source or "api", body.text, chunks)

    store = VectorStore(business_id, _kb_base(request))
    store.add(chunks, vectors, source=body.source or "api")
    store.save()
    return {"doc": doc, "chunks": len(chunks)}


@router.delete("/businesses/{business_id}/kb/docs/{doc_id}", status_code=204)
async def delete_kb_doc(request: Request, business_id: str, doc_id: str) -> None:
    bs = _biz(request)
    doc = bs.get_doc(doc_id)
    if not doc or doc["business_id"] != business_id:
        raise HTTPException(404, "doc not found")
    bs.delete_doc(doc_id)
    # Rebuild vector store from remaining chunks in DB.
    texts = bs.list_chunk_texts(business_id)
    store = VectorStore(business_id, _kb_base(request))
    if not texts:
        # clear on-disk index

        for f in (store.index_path, store.vec_path):
            if f.exists():
                f.unlink()
        return
    embeddings = request.app.state.embeddings
    try:
        vectors = await embeddings.embed(texts)
    except Exception as e:
        raise HTTPException(502, f"re-embed failed: {e}") from e
    store.__init__(business_id, _kb_base(request))  # reset
    store.add(texts, vectors, source="rebuild")
    store.save()


# --- sessions + analytics ---


@router.get("/sessions")
def list_sessions(request: Request, business_id: str | None = None, limit: int = 50) -> dict:
    mem = request.app.state.memory
    return {"sessions": mem.list_sessions(business_id=business_id, limit=min(limit, 500))}


@router.get("/sessions/{call_id}")
def get_session(request: Request, call_id: str) -> dict:
    mem = request.app.state.memory
    sess = mem.get_session(call_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return {"session": sess}


@router.get("/analytics")
def analytics(request: Request, business_id: str | None = None) -> dict:
    mem = request.app.state.memory
    bs = _biz(request)
    total = mem.count_sessions(business_id=business_id)
    sessions = mem.list_sessions(business_id=business_id, limit=500)
    n_with_end = sum(1 for s in sessions if s.get("ended_at"))
    durations = [
        s["ended_at"] - s["started_at"]
        for s in sessions
        if s.get("ended_at") and s.get("started_at")
    ]
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    biz_count = len(bs.list_businesses())
    return {
        "business_id": business_id,
        "calls_total": total,
        "calls_completed": n_with_end,
        "avg_call_seconds": round(avg_duration, 2),
        "businesses_total": biz_count,
    }