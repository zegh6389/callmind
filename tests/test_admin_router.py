import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from callmind.gateway.app import app


class FakeEmbeddings:
    def __init__(self, dim: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1 * (i + 1) for _ in range(self.dim)] for i, _ in enumerate(texts)]

    async def close(self) -> None:
        pass


@asynccontextmanager
async def _setup(tmp_path, monkeypatch):
    db = tmp_path / "callmind.db"
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", str(db))
    monkeypatch.setenv("CALLMIND_KB_DIR", str(tmp_path / "kb"))

    app.state.settings = type("S", (), {})()
    app.state.settings.kb_dir = str(tmp_path / "kb")
    app.state.embeddings = FakeEmbeddings()
    from callmind.admin.store import BusinessStore
    from callmind.brain.memory import MemoryStore

    app.state.business_store = BusinessStore(str(db))
    app.state.memory = MemoryStore(str(db))

    try:
        yield
    finally:
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    asyncio.run(_setup_inner(tmp_path, monkeypatch))
    return TestClient(app)


async def _setup_inner(tmp_path, monkeypatch):
    db = tmp_path / "callmind.db"
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", str(db))
    monkeypatch.setenv("CALLMIND_KB_DIR", str(tmp_path / "kb"))

    from callmind.admin.store import BusinessStore
    from callmind.brain.memory import MemoryStore

    app.state.settings = type("S", (), {})()
    app.state.settings.kb_dir = str(tmp_path / "kb")
    app.state.embeddings = FakeEmbeddings()
    app.state.business_store = BusinessStore(str(db))
    app.state.memory = MemoryStore(str(db))


def test_businesses_crud(client):
    r = client.post("/admin/businesses", json={"name": "Acme"})
    assert r.status_code == 201
    bid = r.json()["business"]["id"]

    r = client.get(f"/admin/businesses/{bid}")
    assert r.status_code == 200
    assert r.json()["business"]["name"] == "Acme"

    r = client.patch(f"/admin/businesses/{bid}", json={"greeting": "Hello!"})
    assert r.json()["business"]["greeting"] == "Hello!"

    r = client.delete(f"/admin/businesses/{bid}")
    assert r.status_code == 204

    r = client.get(f"/admin/businesses/{bid}")
    assert r.status_code == 404


def test_kb_doc_create_and_delete(client):
    r = client.post("/admin/businesses", json={"name": "Acme"})
    bid = r.json()["business"]["id"]

    r = client.post(
        f"/admin/businesses/{bid}/kb/docs",
        json={"source": "faq", "text": "Hello world. " * 50},
    )
    assert r.status_code == 201
    doc_id = r.json()["doc"]["id"]
    assert r.json()["chunks"] > 0

    r = client.get(f"/admin/businesses/{bid}/kb/docs")
    assert len(r.json()["docs"]) == 1

    r = client.delete(f"/admin/businesses/{bid}/kb/docs/{doc_id}")
    assert r.status_code == 204

    r = client.get(f"/admin/businesses/{bid}/kb/docs")
    assert r.json()["docs"] == []


def test_sessions_and_analytics(client):
    r = client.post("/admin/businesses", json={"name": "Acme"})
    bid = r.json()["business"]["id"]

    mem = app.state.memory
    mem.start_conversation("c1", bid, "+15551111111")
    mem.append_message("c1", "user", "hi")
    mem.end_conversation("c1")

    r = client.get("/admin/sessions", params={"business_id": bid})
    assert r.status_code == 200
    assert len(r.json()["sessions"]) == 1

    r = client.get("/admin/sessions/c1")
    assert r.status_code == 200
    assert r.json()["session"]["messages"][0]["content"] == "hi"

    r = client.get("/admin/analytics", params={"business_id": bid})
    body = r.json()
    assert body["calls_total"] == 1
    assert body["calls_completed"] == 1
    assert body["businesses_total"] >= 1