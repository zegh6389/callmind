import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from callmind.gateway.app import app

TOKEN = {"x-admin-token": "test-token"}


class FakeEmbeddings:
    def __init__(self, dim: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        # Stable vector per text so different docs/chunks match distinctly.
        return [
            [
                (0.1 * (i + 1)) + (sum(ord(c) % 7 for c in t) * 0.001)
                for _ in range(self.dim)
            ]
            for i, t in enumerate(texts)
        ]

    async def close(self) -> None:
        pass


@asynccontextmanager
async def _setup(tmp_path, monkeypatch):
    db = tmp_path / "callmind.db"
    monkeypatch.setenv("CALLMIND_MEMORY_DB_PATH", str(db))
    monkeypatch.setenv("CALLMIND_KB_DIR", str(tmp_path / "kb"))

    app.state.settings = type("S", (), {})()
    app.state.settings.kb_dir = str(tmp_path / "kb")
    app.state.settings.admin_token = "test-token"
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
    app.state.settings.admin_token = "test-token"
    app.state.embeddings = FakeEmbeddings()
    app.state.business_store = BusinessStore(str(db))
    app.state.memory = MemoryStore(str(db))


def test_businesses_crud(client):
    r = client.post("/admin/businesses", headers=TOKEN, json={"name": "Acme"})
    assert r.status_code == 201
    bid = r.json()["business"]["id"]

    r = client.get(f"/admin/businesses/{bid}", headers=TOKEN)
    assert r.status_code == 200
    assert r.json()["business"]["name"] == "Acme"

    r = client.patch(f"/admin/businesses/{bid}", headers=TOKEN, json={"greeting": "Hello!"})
    assert r.json()["business"]["greeting"] == "Hello!"

    r = client.delete(f"/admin/businesses/{bid}", headers=TOKEN)
    assert r.status_code == 204

    r = client.get(f"/admin/businesses/{bid}", headers=TOKEN)
    assert r.status_code == 404


def test_kb_doc_create_and_delete(client):
    r = client.post("/admin/businesses", headers=TOKEN, json={"name": "Acme"})
    bid = r.json()["business"]["id"]

    r = client.post(
        f"/admin/businesses/{bid}/kb/docs",
        headers=TOKEN,
        json={"source": "faq", "text": "Hello world. " * 50},
    )
    assert r.status_code == 201
    doc_id = r.json()["doc"]["id"]
    assert r.json()["chunks"] > 0

    r = client.get(f"/admin/businesses/{bid}/kb/docs", headers=TOKEN)
    assert len(r.json()["docs"]) == 1

    r = client.delete(f"/admin/businesses/{bid}/kb/docs/{doc_id}", headers=TOKEN)
    assert r.status_code == 204

    r = client.get(f"/admin/businesses/{bid}/kb/docs", headers=TOKEN)
    assert r.json()["docs"] == []


def test_kb_after_delete_keeps_other_docs_searchable(client, tmp_path):
    from callmind.brain.rag import VectorStore

    r = client.post("/admin/businesses", headers=TOKEN, json={"name": "Acme"})
    bid = r.json()["business"]["id"]

    r = client.post(
        f"/admin/businesses/{bid}/kb/docs",
        headers=TOKEN,
        json={"source": "pricing", "text": "Widgets cost eleven dollars. " * 40},
    )
    doc_a = r.json()["doc"]["id"]
    r = client.post(
        f"/admin/businesses/{bid}/kb/docs",
        headers=TOKEN,
        json={"source": "hours", "text": "We open at nine o'clock. " * 40},
    )

    r = client.delete(f"/admin/businesses/{bid}/kb/docs/{doc_a}", headers=TOKEN)
    assert r.status_code == 204

    store = VectorStore(bid, str(tmp_path / "kb"))
    assert not store.is_empty()
    hits = store.search([0.3, 0.2, 0.1, 0.4], top_k=2)
    texts = [h[0] for h in hits]
    assert all("nine" in t for t in texts)
    assert all("widgets" not in t for t in texts)


def test_sessions_and_analytics(client):
    r = client.post("/admin/businesses", headers=TOKEN, json={"name": "Acme"})
    bid = r.json()["business"]["id"]

    mem = app.state.memory
    mem.start_conversation("c1", bid, "+15551111111")
    mem.append_message("c1", "user", "hi")
    mem.end_conversation("c1")

    r = client.get("/admin/sessions", headers=TOKEN, params={"business_id": bid})
    assert r.status_code == 200
    assert len(r.json()["sessions"]) == 1

    r = client.get("/admin/sessions/c1", headers=TOKEN)
    assert r.status_code == 200
    assert r.json()["session"]["messages"][0]["content"] == "hi"

    r = client.get("/admin/analytics", headers=TOKEN, params={"business_id": bid})
    body = r.json()
    assert body["calls_total"] == 1
    assert body["calls_completed"] == 1
    assert body["businesses_total"] >= 1
