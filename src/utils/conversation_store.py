"""
agentY – self-contained conversation persistence (SQLite).

Replaces the Chainlit SQLAlchemy/Postgres datalayer + MinIO file store. A single
local SQLite file holds every thread, its messages, its generated-image gallery,
and the per-thread pipeline state (compressed Brain history, AgentSession,
last brainbriefing, last prior summary) so a thread can be resumed exactly like
``on_chat_resume`` did under Chainlit — but with **no Docker, Postgres, or S3**.

The store is deliberately dependency-free (stdlib ``sqlite3`` only) and safe for
the threaded Flask bridge: every call opens its own short-lived connection.

Schema
------
threads       (id, title, created_at, updated_at)
messages      (id, thread_id, role, content, created_at)          role: user|assistant|system
gallery       (id, thread_id, idx, path, caption, created_at)
thread_state  (thread_id, brain_messages, agent_session,
               last_brainbriefing, last_prior_summary, updated_at)   JSON text columns
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _db_path() -> Path:
    """Resolve the SQLite file path (settings.json > env > default).

    Priority:
      1. ``AGENTY_CONVERSATION_DB`` env var
      2. ``conversation_db`` in config/settings.json
      3. ``<project_root>/memory/conversations.sqlite``
    """
    env = os.environ.get("AGENTY_CONVERSATION_DB")
    if env:
        return Path(env).expanduser()
    rel = "./memory/conversations.sqlite"
    try:
        cfg_path = _project_root() / "config" / "settings.json"
        if cfg_path.exists():
            cfg = json.loads(
                "".join(ln for ln in cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)
                        if not ln.lstrip().startswith("//"))
            )
            rel = cfg.get("conversation_db", rel)
    except Exception:
        pass
    p = Path(rel)
    return p if p.is_absolute() else (_project_root() / p)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


_INITIALISED = False


def init_db() -> None:
    """Create tables if they don't exist. Idempotent; cheap to call repeatedly."""
    global _INITIALISED
    if _INITIALISED:
        return
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New chat',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
            CREATE TABLE IF NOT EXISTS gallery (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                idx         INTEGER NOT NULL,
                path        TEXT NOT NULL,
                caption     TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gallery_thread ON gallery(thread_id, idx);
            CREATE TABLE IF NOT EXISTS thread_state (
                thread_id           TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
                brain_messages      TEXT,
                agent_session       TEXT,
                last_brainbriefing  TEXT,
                last_prior_summary  TEXT,
                updated_at          REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS thread_panel (
                thread_id   TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
                html        TEXT NOT NULL,
                updated_at  REAL NOT NULL
            );
            """
        )
    _INITIALISED = True


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def create_thread(title: str = "New chat", thread_id: Optional[str] = None) -> str:
    init_db()
    tid = thread_id or uuid.uuid4().hex
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO threads(id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (tid, title or "New chat", now, now),
        )
    return tid


def list_threads(limit: int = 200) -> list[dict[str, Any]]:
    """Return threads newest-first with a message count."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.title, t.created_at, t.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.thread_id = t.id) AS message_count
            FROM threads t
            ORDER BY t.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_thread(thread_id: str) -> Optional[dict[str, Any]]:
    """Return a thread with its ordered messages and gallery, or None."""
    init_db()
    with _connect() as conn:
        head = conn.execute("SELECT * FROM threads WHERE id=?", (thread_id,)).fetchone()
        if head is None:
            return None
        msgs = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE thread_id=? ORDER BY id",
            (thread_id,),
        ).fetchall()
        gal = conn.execute(
            "SELECT idx, path, caption, created_at FROM gallery WHERE thread_id=? ORDER BY idx",
            (thread_id,),
        ).fetchall()
    return {
        **dict(head),
        "messages": [dict(m) for m in msgs],
        "gallery": [dict(g) for g in gal],
    }


def rename_thread(thread_id: str, title: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE threads SET title=?, updated_at=? WHERE id=?",
            (title, time.time(), thread_id),
        )


def delete_thread(thread_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM threads WHERE id=?", (thread_id,))


def delete_all_threads(except_id: Optional[str] = None) -> int:
    """Delete every thread (optionally keeping *except_id*). Returns count deleted."""
    init_db()
    with _connect() as conn:
        if except_id:
            cur = conn.execute("DELETE FROM threads WHERE id<>?", (except_id,))
        else:
            cur = conn.execute("DELETE FROM threads")
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(thread_id: str, role: str, content: str) -> int:
    """Append a message; bumps the thread's updated_at and auto-titles it.

    The first non-empty user message becomes the thread title (first ~60 chars)
    when the thread still carries the default title.
    """
    init_db()
    now = time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages(thread_id, role, content, created_at) VALUES (?,?,?,?)",
            (thread_id, role, content, now),
        )
        conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (now, thread_id))
        if role == "user" and content.strip():
            head = conn.execute("SELECT title FROM threads WHERE id=?", (thread_id,)).fetchone()
            if head is not None and (head["title"] in ("", "New chat")):
                title = " ".join(content.strip().split())[:60]
                conn.execute("UPDATE threads SET title=? WHERE id=?", (title, thread_id))
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Gallery (generated outputs)
# ---------------------------------------------------------------------------

def add_gallery_image(thread_id: str, path: str, caption: str = "") -> int:
    """Register a generated image/output for a thread; idx is 1-based per thread."""
    init_db()
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(idx), 0) AS mx FROM gallery WHERE thread_id=?", (thread_id,)
        ).fetchone()
        idx = int(row["mx"]) + 1
        conn.execute(
            "INSERT INTO gallery(thread_id, idx, path, caption, created_at) VALUES (?,?,?,?,?)",
            (thread_id, idx, path, caption, now),
        )
    return idx


def get_gallery(thread_id: str) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT idx, path, caption, created_at FROM gallery WHERE thread_id=? ORDER BY idx",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Per-thread pipeline state (for resume)
# ---------------------------------------------------------------------------

def save_state(
    thread_id: str,
    *,
    brain_messages: Any = None,
    agent_session: Any = None,
    last_brainbriefing: Optional[str] = None,
    last_prior_summary: Optional[str] = None,
) -> None:
    """Persist the pipeline snapshot for *thread_id* (JSON-encoded)."""
    init_db()
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO thread_state(thread_id, brain_messages, agent_session,
                                     last_brainbriefing, last_prior_summary, updated_at)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(thread_id) DO UPDATE SET
                brain_messages=excluded.brain_messages,
                agent_session=excluded.agent_session,
                last_brainbriefing=excluded.last_brainbriefing,
                last_prior_summary=excluded.last_prior_summary,
                updated_at=excluded.updated_at
            """,
            (
                thread_id,
                json.dumps(brain_messages) if brain_messages is not None else None,
                json.dumps(agent_session) if agent_session is not None else None,
                last_brainbriefing,
                last_prior_summary,
                now,
            ),
        )


def load_state(thread_id: str) -> Optional[dict[str, Any]]:
    """Return the decoded pipeline snapshot for *thread_id*, or None if unsaved."""
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM thread_state WHERE thread_id=?", (thread_id,)).fetchone()
    if row is None:
        return None
    def _load(col: str) -> Any:
        raw = row[col]
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None
    return {
        "brain_messages": _load("brain_messages"),
        "agent_session": _load("agent_session"),
        "last_brainbriefing": row["last_brainbriefing"],
        "last_prior_summary": row["last_prior_summary"],
    }


# ---------------------------------------------------------------------------
# Rendered chat panel (for restoring the exact UI — collapsible think/step
# blocks and all — when a thread is reopened, incl. after a page reload)
# ---------------------------------------------------------------------------

def save_panel(thread_id: str, html: str) -> None:
    """Persist the thread's rendered chat-panel HTML (the live DOM, including
    collapsible think/step blocks), so reopening the thread restores it exactly
    rather than rebuilding from the text-only message log."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO thread_panel(thread_id, html, updated_at) VALUES (?,?,?)
            ON CONFLICT(thread_id) DO UPDATE SET html=excluded.html, updated_at=excluded.updated_at
            """,
            (thread_id, html or "", time.time()),
        )


def get_panel(thread_id: str) -> Optional[str]:
    """Return the saved rendered panel HTML for *thread_id*, or None."""
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT html FROM thread_panel WHERE thread_id=?", (thread_id,)).fetchone()
    return row["html"] if row is not None else None
