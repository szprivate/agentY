"""
agentY – Local FAISS memory layer using mem0 + nomic-embed-text.

All storage is fully local — no external API calls:
  • Vector store : FAISS (persisted to ./memory/ on disk)
  • Embeddings   : nomic-embed-text via Ollama (768-dim)
  • Fact-extract : Ollama (same LLM as llm_functions) for mem0's internal
                   deduplication/extraction pipeline

Public API
----------
>>> from src.utils.memory import memory_search, memory_add, format_memories
>>> memory_add("User prefers 1024×1024 for portrait shots.", session_id="abc")
>>> hits = memory_search("portrait resolution", session_id="abc")
>>> print(format_memories(hits))
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal settings reader (avoids circular-import with src.agent)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.json"
_MEMORY_DIR = _PROJECT_ROOT / "memory"

# Single shared namespace for all durable long-term memory. Explicit writes
# (memory_write), the learnings agent, the trimmed request log, and retrieval all
# target this one ``user_id`` so everything the agent knows lives in one place and
# is recalled together — rather than being fragmented across per-session ids where
# each turn could only see its own writes. Kept as "learnings_global" (its historic
# name) so the lessons already stored under that id stay in place without a
# migration; it is now the home for every memory kind, not just learnings.
MEMORY_NAMESPACE = "learnings_global"

_settings_cache: dict = {}
_settings_lock = threading.Lock()


def _load_settings() -> dict:
    global _settings_cache
    if _settings_cache:
        return _settings_cache
    with _settings_lock:
        if _settings_cache:
            return _settings_cache
        if _SETTINGS_PATH.exists():
            try:
                _settings_cache = json.loads("".join(ln for ln in _SETTINGS_PATH.read_text(encoding="utf-8").splitlines(keepends=True) if not ln.lstrip().startswith("//")))
            except Exception:
                _settings_cache = {}
        return _settings_cache


def _get(env_var: str, *path: str, default: str = "") -> str:
    """Read: env var > settings.json path > default."""
    val = os.environ.get(env_var)
    if val is not None:
        return val
    node: Any = _load_settings()
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    if node and not isinstance(node, dict):
        return str(node)
    return default


# ---------------------------------------------------------------------------
# mem0 Memory singleton
# ---------------------------------------------------------------------------

_mem0_client: Any = None
_mem0_lock = threading.Lock()


def _is_enabled() -> bool:
    """Return False when MEMORY_ENABLED=false or memory.enabled=false."""
    env = os.environ.get("MEMORY_ENABLED", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    from_settings = str(_get("__never_set__", "memory", "enabled", default="true")).lower()
    return from_settings not in ("0", "false", "no", "off")


def _build_config() -> dict:
    """Return the mem0 MemoryConfig dict sourced from env / settings.json."""
    ollama_host = _get("OLLAMA_HOST", "llm", "ollama", "host", default="http://localhost:11434")
    embed_model = _get("MEMORY_EMBED_MODEL", "memory", "embed_model", default="nomic-embed-text")
    embed_dims = int(_get("MEMORY_EMBED_DIMS", "memory", "embed_model_dims", default="768"))
    # Default extraction LLM to the same lightweight model used for triage/functions
    #  so we reuse a model that is already warm in Ollama's context.
    llm_model = _get(
        "MEMORY_LLM_MODEL", "memory", "llm_model",
        default=_get("LLM_FUNCTIONS_MODEL", "llm", "pipeline", "llm_functions", default="qwen3.5:9b"),
    )
    store_dir = str(
        (_PROJECT_ROOT / _get("MEMORY_STORE_DIR", "memory", "store_dir", default="memory")).resolve()
    )
    history_db = str((_PROJECT_ROOT / "memory" / "history.db").resolve())

    return {
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": "agenty_memory",
                "path": store_dir,
                "embedding_model_dims": embed_dims,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "ollama_base_url": ollama_host,
                "embedding_dims": embed_dims,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "ollama_base_url": ollama_host,
                "temperature": 0.1,
            },
        },
        "history_db_path": history_db,
        "version": "v1.1",
    }


def _ensure_model(model_id: str, host: str) -> None:
    """Pull *model_id* via Ollama if not already present (best-effort)."""
    try:
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}
        normalised = model_id if ":" in model_id else f"{model_id}:latest"
        if normalised in names or model_id in names:
            return
    except Exception:
        pass  # network error → just try to use it anyway

    import subprocess
    try:
        print(f"[memory] Pulling embedding model '{model_id}' via Ollama …")
        subprocess.run(["ollama", "pull", model_id], check=True)
    except Exception as exc:
        print(f"[memory] Warning: could not pull '{model_id}': {exc}")


def mem0_client() -> Any:
    """Return the singleton mem0 Memory instance (lazy, thread-safe init)."""
    global _mem0_client
    if _mem0_client is not None:
        return _mem0_client
    with _mem0_lock:
        if _mem0_client is not None:
            return _mem0_client
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        cfg = _build_config()
        # Ensure the embedding model is available in Ollama before first use.
        embed_model = cfg["embedder"]["config"]["model"]
        ollama_host = cfg["embedder"]["config"]["ollama_base_url"]
        _ensure_model(embed_model, ollama_host)

        from mem0 import Memory
        _mem0_client = Memory.from_config(config_dict=cfg)
        print(f"[memory] FAISS memory layer initialised  (path={cfg['vector_store']['config']['path']},"
              f" embed={embed_model}, llm={cfg['llm']['config']['model']})")
        return _mem0_client


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def memory_search(query: str, session_id: str = "default", limit: int = 5) -> list[dict]:
    """Return up to *limit* memories relevant to *query* for *session_id*.

    Returns an empty list if memory is disabled or on any error.
    """
    if not _is_enabled():
        return []
    try:
        client = mem0_client()
        results = client.search(query, filters={"user_id": session_id}, limit=limit)
        # mem0 may return a dict {"results": [...]} or a plain list
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []
    except Exception as exc:
        print(f"[memory] search error: {exc}")
        return []


def memory_add(
    content: str,
    session_id: str = MEMORY_NAMESPACE,
    metadata: dict | None = None,
    infer: bool = False,
) -> dict | None:
    """Persist *content* to long-term memory for *session_id*.

    ``infer=False`` (the default) stores the sentence **verbatim**. We deliberately
    do NOT route through mem0's LLM fact-extraction/dedup by default: the small
    local extraction model frequently distils *zero* facts from instruction-style
    content (e.g. "Nano Banana Pro and Nano Banana 2 are distinct models — never
    substitute one for the other") and silently drops it, so an explicit "remember
    this" would vanish while the caller was told it was saved. Pass ``infer=True``
    only when you deliberately want mem0 to distil atomic facts from a longer text.

    Returns mem0's add result (``{"results": [{"id", "event", ...}]}``) so callers
    can confirm what actually landed, or ``None`` when memory is disabled or the
    write errored. Best-effort — never raises.
    """
    if not _is_enabled():
        return None
    try:
        client = mem0_client()
        return client.add(content, user_id=session_id, metadata=metadata or {}, infer=infer)
    except Exception as exc:
        print(f"[memory] add error: {exc}")
        return None


def memory_get_all(session_id: str = "default") -> list[dict]:
    """Return all stored memories for *session_id*."""
    if not _is_enabled():
        return []
    try:
        client = mem0_client()
        results = client.get_all(user_id=session_id)
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []
    except Exception as exc:
        print(f"[memory] get_all error: {exc}")
        return []


def format_memories(memories: list[dict], header: str = "## Relevant memories from past sessions") -> str:
    """Return a human-readable Markdown block from a list of mem0 result dicts.

    Returns an empty string when *memories* is empty so callers can test
    truthiness before injecting into prompts.
    """
    if not memories:
        return ""
    lines = [header, ""]
    for m in memories:
        text = m.get("memory") or m.get("text") or str(m)
        score = m.get("score")
        score_hint = f" (relevance: {score:.2f})" if score is not None else ""
        lines.append(f"- {text}{score_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Admin helpers — power the long-term-memory viewer (list / edit / delete /
# purge). Listing reads the on-disk FAISS docstore directly so the viewer needs
# neither Ollama nor a live client and can show memories across every session;
# mutations go through the mem0 client so the vector index stays consistent.
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "agenty_memory"


def _store_path() -> Path:
    """Path to the FAISS collection JSON on disk (matches ``_build_config``)."""
    store_dir = _get("MEMORY_STORE_DIR", "memory", "store_dir", default="memory")
    return (_PROJECT_ROOT / store_dir).resolve() / f"{_COLLECTION_NAME}.json"


def memory_list_raw() -> list[dict]:
    """Return every stored long-term memory straight from the FAISS docstore.

    Reads the collection JSON directly, so it needs neither Ollama nor a live
    mem0 client and lists memories across every ``user_id``/session. Each item is
    ``{id, text, user_id, created_at, updated_at, attributed_to, hash}``, sorted
    newest first. Returns ``[]`` when the store doesn't exist yet or on any error.
    """
    path = _store_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace") or "{}")
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] list_raw parse error: {exc}")
        return []
    docstore = raw.get("docstore") if isinstance(raw, dict) else None
    if not isinstance(docstore, dict):
        return []
    out: list[dict] = []
    for mid, payload in docstore.items():
        if not isinstance(payload, dict):
            continue
        out.append({
            "id": mid,
            "text": payload.get("data") or payload.get("memory") or "",
            "user_id": payload.get("user_id"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "attributed_to": payload.get("attributed_to"),
            "hash": payload.get("hash"),
        })
    # Newest first; ISO-8601 timestamps sort correctly as plain strings.
    out.sort(key=lambda m: (m.get("updated_at") or m.get("created_at") or ""), reverse=True)
    return out


def memory_update(memory_id: str, text: str) -> None:
    """Overwrite a single memory's text (re-embeds via the mem0 client)."""
    mem0_client().update(memory_id, text)


def memory_delete_ids(ids: list[str]) -> dict:
    """Delete each memory id via the mem0 client.

    Returns ``{"deleted": n, "errors": [{"id", "error"}, …]}`` so the caller can
    report partial failures instead of aborting the whole batch on the first bad id.
    """
    client = mem0_client()
    deleted, errors = 0, []
    for mid in ids or []:
        try:
            client.delete(mid)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": mid, "error": str(exc)})
    return {"deleted": deleted, "errors": errors}


def memory_purge() -> dict:
    """Delete every stored memory across all sessions. Returns delete_ids' report."""
    ids = [m["id"] for m in memory_list_raw()]
    return memory_delete_ids(ids)
