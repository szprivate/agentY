"""
agentY – FAISS memory layer (mem0).

  • Vector store : FAISS (persisted to ./memory/ on disk) — always local.
  • Embeddings   : configurable provider (config/settings.json ▸ memory.embedder).
                   "ollama" (default, fully local, e.g. nomic-embed-text 768-dim)
                   or "openai" — any OpenAI-compatible endpoint incl. DashScope/
                   Qwen — so a host without Ollama can still run memory.
  • Fact-extract : configurable LLM (memory.llm); only used for infer=True writes.

Switching the embedder (or its dims) makes the existing on-disk index
incompatible — purge memory/agenty_memory.* so it rebuilds at the new size.

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

from src.utils.status_bus import notify

# ---------------------------------------------------------------------------
# Minimal settings reader (avoids circular-import with src.agent)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MEMORY_DIR = _PROJECT_ROOT / "memory"

# Single shared namespace for all durable long-term memory. Explicit writes
# (memory_write), the learnings agent, the trimmed request log, and retrieval all
# target this one ``user_id`` so everything the agent knows lives in one place and
# is recalled together — rather than being fragmented across per-session ids where
# each turn could only see its own writes. Kept as "learnings_global" (its historic
# name) so the lessons already stored under that id stay in place without a
# migration; it is now the home for every memory kind, not just learnings.
MEMORY_NAMESPACE = "learnings_global"

def _load_settings() -> dict:
    """Merged settings (TOML defaults ⊕ local JSON overrides) via the app loader."""
    from src.utils.settings import load_settings
    return load_settings()


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

# Every memory op ultimately calls the embedder (Ollama by default). When that
# provider is slow or wedged — e.g. Ollama swapping the embed model into VRAM a
# generation just filled, or an unreachable endpoint — an unbounded ``.embed()``
# blocks its caller forever. The end-of-turn learnings write and the *next*
# turn's recall both embed, so a single wedged call strands the following turn
# ("agent stalls after 'Orchestrator finished', needs restart"). Bounding each
# op means a hung embedder degrades to an empty/None result instead of hanging
# the turn. The orphaned worker unwinds when the embedder returns or the process
# exits (daemon threads).
import concurrent.futures as _futures  # noqa: E402

_MEM_OP_TIMEOUT = float(os.environ.get("AGENTY_MEMORY_OP_TIMEOUT", "20") or 20)
_mem_executor = _futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mem-op")


def _bounded(fn: Any, default: Any, label: str) -> Any:
    """Run blocking memory op *fn* under ``_MEM_OP_TIMEOUT``; return *default* on
    timeout or error so a wedged embedder can never stall the caller."""
    try:
        return _mem_executor.submit(fn).result(timeout=_MEM_OP_TIMEOUT)
    except _futures.TimeoutError:
        print(f"[memory] {label} exceeded {_MEM_OP_TIMEOUT:.0f}s — skipping "
              "(embedder slow or unreachable).")
        return default
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] {label} error: {exc}")
        return default


def _is_enabled() -> bool:
    """Return False when MEMORY_ENABLED=false or memory.enabled=false."""
    env = os.environ.get("MEMORY_ENABLED", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    from_settings = str(_get("__never_set__", "memory", "enabled", default="true")).lower()
    return from_settings not in ("0", "false", "no", "off")


_PROVIDER_ENV = {
    "embedder": {"provider": "MEMORY_EMBEDDER_PROVIDER", "base_url": "MEMORY_EMBEDDER_BASE_URL",
                 "api_key_env": "MEMORY_EMBEDDER_API_KEY_ENV"},
    "llm": {"provider": "MEMORY_LLM_PROVIDER", "base_url": "MEMORY_LLM_BASE_URL",
            "api_key_env": "MEMORY_LLM_API_KEY_ENV"},
}


def _mem_provider(kind: str) -> str:
    """Provider for the memory *embedder* or *llm*: 'ollama' (default, fully local)
    or 'openai' (any OpenAI-compatible endpoint, incl. DashScope/Qwen — no Ollama)."""
    val = _get(_PROVIDER_ENV[kind]["provider"], "memory", kind, "provider", default="ollama")
    return (val or "ollama").strip().lower()


def _inherited_endpoint(provider: str) -> dict | None:
    """Endpoint + key-env for a provider inherited from the pipeline settings.

    When the memory LLM is *not* configured explicitly it borrows the model from the
    llm_functions role, and it has to borrow that provider's endpoint too — sending a
    Qwen model name to api.openai.com with an empty key is not a useful default.
    Returns None when the provider isn't one we can reach OpenAI-compatibly.
    """
    p = (provider or "").strip().lower()
    settings = _load_settings()
    if p in ("dashscope", "qwen", "modelstudio", "alibaba"):
        return {
            "base_url": str(((settings.get("llm") or {}).get("dashscope") or {}).get("base_url") or "").strip(),
            "api_key_env": "DASHSCOPE_API_KEY",
        }
    if p == "openai":
        return {"base_url": "", "api_key_env": "OPENAI_API_KEY"}
    if p in ("google", "gemini"):
        return {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "api_key_env": "GEMINI_API_KEY"}
    return None


def _openai_compat_cfg(kind: str, extra: dict, fallback: dict | None = None) -> dict:
    """mem0 config block for an OpenAI-compatible embedder/llm — real OpenAI when
    base_url is blank, or DashScope/Qwen (etc.) when it points at the compatible-mode
    URL. The key is read from the env var named in settings (default OPENAI_API_KEY),
    so secrets stay in .env and are never persisted to settings.json.

    *fallback* supplies the endpoint inherited from another provider's block when the
    ``memory`` settings leave it blank (see :func:`_inherited_endpoint`)."""
    base_url = _get(_PROVIDER_ENV[kind]["base_url"], "memory", kind, "base_url", default="").strip()
    key_env = (_get(_PROVIDER_ENV[kind]["api_key_env"], "memory", kind, "api_key_env",
                    default="").strip())
    if fallback:
        base_url = base_url or str(fallback.get("base_url") or "").strip()
        key_env = key_env or str(fallback.get("api_key_env") or "").strip()
    key_env = key_env or "OPENAI_API_KEY"
    cfg = dict(extra)
    cfg["api_key"] = os.environ.get(key_env, "")
    if base_url:
        cfg["openai_base_url"] = base_url
    return cfg


def _build_config() -> dict:
    """Return the mem0 MemoryConfig dict sourced from env / settings.json.

    The vector store is always local FAISS. The embedder and the (rarely used)
    fact-extraction LLM each choose their provider independently — "ollama" or
    "openai"-compatible — via the ``memory`` block in config/settings.json, so a
    machine without Ollama can run memory off a key it already has (e.g. DashScope).
    """
    from src.utils.settings import ollama_host as _resolve_ollama_host
    ollama_host = _resolve_ollama_host()

    # ── Embedder (turns text into vectors for FAISS) ─────────────────────────
    embed_provider = _mem_provider("embedder")
    # Legacy flat keys (memory.embed_model / memory.embed_model_dims) still honoured.
    embed_model = _get("MEMORY_EMBED_MODEL", "memory", "embedder", "model",
                       default=_get("__unset__", "memory", "embed_model",
                                    default="nomic-embed-text" if embed_provider == "ollama"
                                    else "text-embedding-v3"))
    embed_dims = int(_get("MEMORY_EMBED_DIMS", "memory", "embedder", "embedding_dims",
                          default=_get("__unset__", "memory", "embed_model_dims",
                                       default="768" if embed_provider == "ollama" else "1024")))
    if embed_provider == "ollama":
        embedder = {"provider": "ollama", "config": {
            "model": embed_model, "ollama_base_url": ollama_host, "embedding_dims": embed_dims}}
    else:
        embedder = {"provider": "openai", "config": _openai_compat_cfg(
            "embedder", {"model": embed_model, "embedding_dims": embed_dims})}

    # ── Fact-extraction LLM (only invoked for infer=True writes) ─────────────
    # With no explicit memory.llm.model, inherit the llm_functions role — resolved
    # through the model tiers, so it follows whatever the rest of the system is
    # actually using. Reading llm.pipeline.llm_functions directly is not enough:
    # per-role entries are blank by default now (they mean "inherit"), which used
    # to drop this to a hard-coded Ollama model nobody had configured.
    inherited_spec = ""
    try:
        from src.agent import role_model
        inherited_spec = role_model("llm_functions", env_var="LLM_FUNCTIONS_MODEL")
    except Exception as exc:  # noqa: BLE001
        notify(f"[memory] could not resolve the llm_functions model ({exc})", level="warning")

    llm_model = _get("MEMORY_LLM_MODEL", "memory", "llm", "model",
                     default=_get("__unset__", "memory", "llm_model", default=""))
    explicit_llm = bool(llm_model.strip())
    if not explicit_llm:
        llm_model = inherited_spec or "qwen3.5:9b"

    # A "provider,model" spec carries the provider too — honour it when inheriting,
    # otherwise a DashScope model would be handed to a local Ollama client.
    inherited_provider = ""
    if "," in llm_model:
        prov, _, mdl = llm_model.partition(",")
        inherited_provider = prov.strip().lower()
        llm_model = mdl.strip()
    llm_provider = _mem_provider("llm")
    if not explicit_llm and inherited_provider and inherited_provider != "ollama":
        # Anything that isn't Ollama reaches mem0 through its OpenAI-compatible path.
        llm_provider = "openai"
    if llm_provider == "ollama":
        llm = {"provider": "ollama", "config": {
            "model": llm_model, "ollama_base_url": ollama_host, "temperature": 0.1}}
    else:
        llm = {"provider": "openai", "config": _openai_compat_cfg(
            "llm", {"model": llm_model, "temperature": 0.1},
            fallback=_inherited_endpoint(inherited_provider) if not explicit_llm else None)}

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
        "embedder": embedder,
        "llm": llm,
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
        notify(f"[memory] Pulling embedding model '{model_id}' via Ollama …")
        # Bounded so a wedged pull (mem0 client init holds _mem0_lock during this)
        # can't hang the first memory op indefinitely.
        subprocess.run(["ollama", "pull", model_id], check=True,
                       timeout=float(os.environ.get("AGENTY_OLLAMA_PULL_TIMEOUT", "600") or 600))
    except Exception as exc:
        notify(f"[memory] Warning: could not pull '{model_id}': {exc}", level="warning")


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
        # Only Ollama needs a local pull; OpenAI-compatible providers are remote.
        if cfg["embedder"]["provider"] == "ollama":
            _ensure_model(cfg["embedder"]["config"]["model"],
                          cfg["embedder"]["config"]["ollama_base_url"])

        from mem0 import Memory
        _mem0_client = Memory.from_config(config_dict=cfg)
        notify(f"[memory] FAISS memory layer initialised  (path={cfg['vector_store']['config']['path']},"
               f" embed={cfg['embedder']['provider']}:{cfg['embedder']['config']['model']},"
               f" llm={cfg['llm']['provider']}:{cfg['llm']['config']['model']})")
        return _mem0_client


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def memory_search(
    query: str,
    session_id: str = MEMORY_NAMESPACE,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """Return up to *limit* memories most relevant to *query* for *session_id*.

    Ranks by cosine similarity directly against the FAISS index rather than going
    through mem0's ``Memory.search``. mem0 2.0.2's hybrid ``score_and_rank`` is
    broken for the FAISS backend: the provider hands it the raw L2 *distance*
    (lower = better) but the scorer treats it as a *similarity* and sorts
    descending, so it returns near-inverted results (the best matches sink, the
    worst float to the top and saturate at score 1.0). We embed the query with
    mem0's own (normalised) embedder, search the index, map hits back through the
    provider's ``index_to_id``/``docstore``, and score cosine = ``1 - d/2`` (unit
    vectors). ``min_score`` drops weak matches (useful for always-on recall).

    Each result is ``{"id", "memory", "score", "user_id"}``, best first. Returns
    ``[]`` if memory is disabled, the store is empty, or on any error.
    """
    if not _is_enabled():
        return []

    def _do() -> list[dict]:
        import numpy as np

        client = mem0_client()
        vs = client.vector_store
        index = getattr(vs, "index", None)
        if index is None or getattr(index, "ntotal", 0) == 0:
            return []

        vec = np.asarray(client.embedding_model.embed(query, "search"), dtype="float32").reshape(1, -1)
        # Over-fetch so per-user filtering still yields up to `limit` hits.
        k = min(int(index.ntotal), max(limit * 5, 20))
        distances, positions = index.search(vec, k)

        out: list[dict] = []
        for dist, pos in zip(distances[0], positions[0]):
            pos = int(pos)
            if pos < 0:
                continue
            vid = vs.index_to_id.get(pos)
            if vid is None:
                vid = vs.index_to_id.get(str(pos))
            payload = vs.docstore.get(vid) if vid is not None else None
            if not isinstance(payload, dict):
                continue
            if session_id and payload.get("user_id") != session_id:
                continue
            text = payload.get("data") or payload.get("memory") or ""
            if not text:
                continue
            # IndexFlatL2 returns squared L2; for unit vectors d = 2 - 2cos.
            score = 1.0 - float(dist) / 2.0
            if score < min_score:
                continue
            out.append({
                "id": vid,
                "memory": text,
                "score": round(score, 4),
                "user_id": payload.get("user_id"),
            })
            if len(out) >= limit:
                break
        return out

    return _bounded(_do, [], "search")


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
    return _bounded(
        lambda: mem0_client().add(content, user_id=session_id,
                                  metadata=metadata or {}, infer=infer),
        None, "add")


def memory_get_all(session_id: str = "default") -> list[dict]:
    """Return all stored memories for *session_id*."""
    if not _is_enabled():
        return []

    def _do() -> list[dict]:
        results = mem0_client().get_all(user_id=session_id)
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    return _bounded(_do, [], "get_all")


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
