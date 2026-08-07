import pytest

from callmind.admin.store import BusinessStore


@pytest.fixture
def store(tmp_path):
    return BusinessStore(str(tmp_path / "admin.db"))


def test_create_and_get(store):
    b = store.create_business(name="Acme")
    assert b["name"] == "Acme"
    assert b["id"]
    fetched = store.get_business(b["id"])
    assert fetched == b


def test_list_businesses(store):
    store.create_business(name="A")
    store.create_business(name="B")
    assert len(store.list_businesses()) == 2


def test_update_business(store):
    b = store.create_business(name="Acme")
    updated = store.update_business(b["id"], name="Acme Inc", greeting="Hi!")
    assert updated["name"] == "Acme Inc"
    assert updated["greeting"] == "Hi!"


def test_delete_business_cascades_docs(store):
    b = store.create_business(name="Acme")
    doc = store.create_doc(b["id"], source="faq", text="Hello world. " * 50, chunks=["a", "b", "c"])
    assert doc["id"]
    assert store.delete_business(b["id"])
    assert store.get_business(b["id"]) is None
    assert store.get_doc(doc["id"]) is None


def test_kb_doc_roundtrip(store):
    b = store.create_business(name="Acme")
    chunks = ["chunk one text", "chunk two text"]
    doc = store.create_doc(b["id"], source="faq", text="...", chunks=chunks)
    listed = store.list_docs(b["id"])
    assert len(listed) == 1
    assert listed[0]["id"] == doc["id"]
    texts = store.list_chunk_texts(b["id"])
    assert texts == chunks


def test_delete_doc(store):
    b = store.create_business(name="Acme")
    doc = store.create_doc(b["id"], source="x", text="...", chunks=["a"])
    assert store.delete_doc(doc["id"])
    assert store.get_doc(doc["id"]) is None