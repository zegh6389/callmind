
import numpy as np

from callmind.brain.rag import VectorStore, chunk_text, cosine


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n   ") == []


def test_chunk_text_short():
    chunks = chunk_text("Hello world.")
    assert chunks == ["Hello world."]


def test_chunk_text_long():
    text = ("The quick brown fox jumps over the lazy dog. " * 50).strip()
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # every chunk is non-empty
    assert all(c.strip() for c in chunks)
    # chunks roughly fit chunk_size (with overlap tolerance)
    assert all(len(c) <= 250 for c in chunks)


def test_vector_store_search(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    chunks = ["hours are 9-5", "we close on Sunday", "free shipping over 50"]
    vecs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0],
    ]
    store.add(chunks, vecs, source="test")
    store.save()

    # query close to "hours" vector
    hits = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0][0] == "hours are 9-5"
    assert hits[0][1] > 0.99

    # persistence: reload and search again
    store2 = VectorStore("biz", str(tmp_path))
    assert len(store2._chunks) == 3
    hits2 = store2.search([0.7, 0.7, 0.0], top_k=1)
    assert hits2[0][0] == "free shipping over 50"


def test_vector_store_empty_search(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    assert store.search([1.0, 0.0], top_k=3) == []


def test_vector_store_zero_query_returns_empty_not_nan(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    store.add(["hours are 9 to 5"], [[1.0, 0.0, 0.0]], source="t")
    hits = store.search([0.0, 0.0, 0.0], top_k=1)
    assert hits == []
    import math

    assert not any(isinstance(h[1], float) and math.isnan(h[1]) for h in [])


def test_vector_store_reset_clears_rows(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], source="t")
    assert not store.is_empty()
    store.reset()
    assert store.is_empty()
    assert store.search([1.0, 0.0], top_k=2) == []
    # reset must clear both state and vectors, so the store remains usable.
    store.add(["c"], [[0.0, 1.0]], source="t")
    assert [h[0] for h in store.search([0.0, 1.0], top_k=1)] == ["c"]


def test_vector_store_save_atomic_writes_and_replaces(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    store.add(["a"], [[1.0, 0.0]], source="t")
    store.save_atomic()
    assert store.index_path.exists()
    assert store.vec_path.exists()
    # No stray temp files left behind.
    assert not store.index_path.with_suffix(".json.tmp").exists()
    assert not list(tmp_path.glob("**/*.tmp*"))


def test_vector_store_save_atomic_empty_drops_stale_vectors(tmp_path):
    store = VectorStore("biz", str(tmp_path))
    store.add(["a"], [[1.0, 0.0]], source="t")
    store.save_atomic()
    assert store.vec_path.exists()
    # Simulate clearing in-memory state (e.g. delete-kb-doc with no survivors).
    store.reset()
    store.save_atomic()
    assert store.index_path.exists()
    assert not store.vec_path.exists()
    # Reload from disk and verify no stale vectors come back.
    reloaded = VectorStore("biz", str(tmp_path))
    assert reloaded.is_empty()
    assert reloaded.search([1.0, 0.0], top_k=2) == []
    # And the store is still usable after the empty snapshot.
    reloaded.add(["c"], [[0.0, 1.0]], source="t")
    assert [h[0] for h in reloaded.search([0.0, 1.0], top_k=1)] == ["c"]


def test_vector_store_search_zero_row_corpus_returns_empty(tmp_path):
    import math

    # A row with zero vector in the corpus would produce NaN scores; we
    # filter them out rather than return garbage.
    store = VectorStore("biz", str(tmp_path))
    store.add(["bad", "good"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], source="t")
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [h[0] for h in hits] == ["good"]
    for _, score, _ in hits:
        assert not math.isnan(score)


def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert abs(cosine(a, a) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine(a, b)) < 1e-6


def test_cosine_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine(a, b) == 0.0