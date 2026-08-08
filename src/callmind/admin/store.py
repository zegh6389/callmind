from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS businesses (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    greeting TEXT NOT NULL DEFAULT '',
    escalation_confidence REAL NOT NULL DEFAULT 0.55,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_docs (
    id TEXT PRIMARY KEY,
    business_id TEXT NOT NULL,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kb_docs_biz ON kb_docs(business_id);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    business_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES kb_docs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_biz ON kb_chunks(business_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(doc_id);
"""


class BusinessStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # --- businesses ---
    def list_businesses(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM businesses ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_business(self, business_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute("SELECT * FROM businesses WHERE id = ?", (business_id,)).fetchone()
        return dict(r) if r else None

    def create_business(self, name: str, prompt: str = "", greeting: str = "") -> dict:
        bid = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO businesses(id, name, prompt, greeting, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (bid, name, prompt, greeting, now, now),
            )
            conn.commit()
        return self.get_business(bid)

    def update_business(self, business_id: str, **fields) -> dict | None:
        allowed = {"name", "prompt", "greeting", "escalation_confidence"}
        sets: list[str] = []
        values: list = []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                values.append(v)
        if not sets:
            return self.get_business(business_id)
        sets.append("updated_at = ?")
        values.append(time.time())
        values.append(business_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE businesses SET {', '.join(sets)} WHERE id = ?", values)
            conn.commit()
        return self.get_business(business_id)

    def delete_business(self, business_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM businesses WHERE id = ?", (business_id,))
            conn.commit()
        return cur.rowcount > 0

    # --- KB docs ---
    def list_docs(self, business_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source, created_at, length(text) AS length FROM kb_docs WHERE business_id = ? ORDER BY created_at DESC",
                (business_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_doc(self, doc_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute("SELECT * FROM kb_docs WHERE id = ?", (doc_id,)).fetchone()
        return dict(r) if r else None

    def create_doc(self, business_id: str, source: str, text: str, chunks: list[str]) -> dict:
        doc_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO kb_docs(id, business_id, source, text, created_at) VALUES (?,?,?,?,?)",
                (doc_id, business_id, source, text, now),
            )
            for i, ch in enumerate(chunks):
                cid = uuid.uuid4().hex[:16]
                conn.execute(
                    "INSERT INTO kb_chunks(id, doc_id, business_id, chunk_index, text) VALUES (?,?,?,?,?)",
                    (cid, doc_id, business_id, i, ch),
                )
            conn.commit()
        return self.get_doc(doc_id)

    def delete_doc(self, doc_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM kb_docs WHERE id = ?", (doc_id,))
            conn.commit()
        return cur.rowcount > 0

    def list_chunks_for_business(
        self, business_id: str, exclude_doc_id: str | None = None
    ) -> list[tuple[str, str]]:
        """Return (chunk_id, text) for all chunks of a business. Used by KB rebuild."""
        with self._lock, self._connect() as conn:
            if exclude_doc_id is None:
                rows = conn.execute(
                    "SELECT id, text FROM kb_chunks WHERE business_id = ? ORDER BY doc_id, chunk_index",
                    (business_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, text FROM kb_chunks WHERE business_id = ? AND doc_id != ? "
                    "ORDER BY doc_id, chunk_index",
                    (business_id, exclude_doc_id),
                ).fetchall()
        return [(r["id"], r["text"]) for r in rows]

    def list_chunk_texts(
        self, business_id: str, exclude_doc_id: str | None = None
    ) -> list[str]:
        return [t for _, t in self.list_chunks_for_business(business_id, exclude_doc_id)]