from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("callmind.memory")


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    call_id TEXT PRIMARY KEY,
    business_id TEXT,
    caller_phone TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_caller
    ON conversations(business_id, caller_phone, started_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    FOREIGN KEY (call_id) REFERENCES conversations(call_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_call
    ON messages(call_id, ts);
"""


class MemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def start_conversation(
        self,
        call_id: str,
        business_id: str | None,
        caller_phone: str | None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations(call_id, business_id, caller_phone, started_at) VALUES (?,?,?,?)",
                (call_id, business_id, caller_phone, time.time()),
            )
            conn.commit()

    def end_conversation(self, call_id: str, summary: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET ended_at = ?, summary = COALESCE(?, summary) WHERE call_id = ?",
                (time.time(), summary, call_id),
            )
            conn.commit()

    def append_message(self, call_id: str, role: str, content: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(call_id, role, content, ts) VALUES (?,?,?,?)",
                (call_id, role, content, time.time()),
            )
            conn.commit()

    def load_recent(
        self,
        business_id: str | None,
        caller_phone: str | None,
        limit: int,
    ) -> list[tuple[str, str]]:
        """Return up to `limit` recent (role, content) messages from prior calls."""
        if not caller_phone:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.role, m.content
                FROM messages m
                JOIN conversations c ON c.call_id = m.call_id
                WHERE c.caller_phone = ?
                  AND (? IS NULL OR c.business_id = ?)
                ORDER BY m.ts DESC
                LIMIT ?
                """,
                (caller_phone, business_id, business_id, limit),
            ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def list_sessions(self, business_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            if business_id:
                rows = conn.execute(
                    "SELECT call_id, business_id, caller_phone, started_at, ended_at, summary FROM conversations WHERE business_id = ? ORDER BY started_at DESC LIMIT ?",
                    (business_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT call_id, business_id, caller_phone, started_at, ended_at, summary FROM conversations ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, call_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            r = conn.execute(
                "SELECT call_id, business_id, caller_phone, started_at, ended_at, summary FROM conversations WHERE call_id = ?",
                (call_id,),
            ).fetchone()
            if not r:
                return None
            sess = dict(r)
            msgs = conn.execute(
                "SELECT role, content, ts FROM messages WHERE call_id = ? ORDER BY ts",
                (call_id,),
            ).fetchall()
        sess["messages"] = [dict(m) for m in msgs]
        return sess

    def count_sessions(self, business_id: str | None = None) -> int:
        with self._lock, self._connect() as conn:
            if business_id:
                r = conn.execute(
                    "SELECT COUNT(*) AS n FROM conversations WHERE business_id = ?", (business_id,)
                ).fetchone()
            else:
                r = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()
        return int(r["n"])