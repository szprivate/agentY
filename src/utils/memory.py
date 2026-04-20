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
        results = client.search(query, user_id=session_id, limit=limit)
        # mem0 may return a dict {"results": [...]} or a plain list
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []
    except Exception as exc:
        print(f"[memory] search error: {exc}")
        return []


def memory_add(content: str, session_id: str = "default", metadata: dict | None = None) -> None:
    """Persist *content* as a memory for *session_id*.

    mem0 internally extracts atomic facts via the configured Ollama LLM,
    deduplicates against existing memories, and stores the embeddings in FAISS.
    This is best-effort — any error is logged and silently swallowed.
    """
    if not _is_enabled():
        return
    try:
        client = mem0_client()
        client.add(content, user_id=session_id, metadata=metadata or {})
    except Exception as exc:
        print(f"[memory] add error: {exc}")


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
