
import pytest

from callmind.brain.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    return MemoryStore(str(db))


def test_start_and_append(store):
    store.start_conversation("c1", business_id="b1", caller_phone="+15551111111")
    store.append_message("c1", "user", "hi")
    store.append_message("c1", "assistant", "hello there")
    store.append_message("c1", "user", "how are you?")


def test_recent_messages_excludes_other_business(store):
    store.start_conversation("c1", "b1", "+15551111111")
    store.append_message("c1", "user", "alpha")
    store.start_conversation("c2", "b2", "+15551111111")
    store.append_message("c2", "user", "beta")

    recent = store.load_recent("b1", "+15551111111", limit=10)
    contents = [c for _, c in recent]
    assert "alpha" in contents
    assert "beta" not in contents


def test_recent_messages_no_phone(store):
    assert store.load_recent("b1", None, limit=10) == []


def test_end_conversation(store):
    store.start_conversation("c1", "b1", "+15551111111")
    store.end_conversation("c1", summary="a polite greeting")


def test_append_message_requires_existing_call(store):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.append_message("ghost-call", "user", "hi")