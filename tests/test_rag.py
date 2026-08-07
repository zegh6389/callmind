
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