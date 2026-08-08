from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("callmind.rag")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n and (nxt := text.find(". ", end)) != -1 and nxt - end < 80:
            end = nxt + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VectorStore:
    """Per-business on-disk vector store. JSON index, npz vectors.

    Simple and dependency-free. Fine for v1 (KBs < 10k chunks)."""

    def __init__(self, business_id: str, base_dir: str) -> None:
        self.business_id = business_id
        self.dir = Path(base_dir) / business_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.vec_path = self.dir / "vectors.npz"
        self._chunks: list[dict] = []
        self._vectors: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                self._chunks = json.load(f)
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path)["v"]

    def save(self) -> None:
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        if self._vectors is not None and self._vectors.size:
            np.savez_compressed(self.vec_path, v=self._vectors)
        elif self.vec_path.exists():
            self.vec_path.unlink()

    def reset(self) -> None:
        """Drop all in-memory chunks and vectors; on-disk index untouched."""
        self._chunks = []
        self._vectors = None

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def add(self, chunks: list[str], vectors: list[list[float]], source: str = "") -> None:
        if not chunks:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(chunks):
            raise ValueError("vectors must align with chunks")
        start_id = len(self._chunks)
        for i, ch in enumerate(chunks):
            self._chunks.append({"id": start_id + i, "text": ch, "source": source})
        if self._vectors is None:
            self._vectors = arr
        else:
            self._vectors = np.vstack([self._vectors, arr])

    def search(self, query_vec: list[float], top_k: int = 3) -> list[tuple[str, float, str]]:
        if self._vectors is None or not self._chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        row_norms = np.linalg.norm(self._vectors, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = (self._vectors @ q) / (row_norms * q_norm + 1e-12)
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        # Drop rows whose stored vector is itself zero (no signal to match).
        scores = np.where(row_norms > 0, scores, -np.inf)
        order = np.argsort(-scores)
        out: list[tuple[str, float, str]] = []
        for i in order:
            if scores[i] == -np.inf:
                continue
            if len(out) >= top_k:
                break
            out.append(
                (
                    self._chunks[i]["text"],
                    float(scores[i]),
                    self._chunks[i].get("source", ""),
                )
            )
        return out