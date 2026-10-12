"""A local model gives the GPU back before ComfyUI renders.

Ollama keeps a model loaded for minutes after its last answer. On a machine
where the agent's model and ComfyUI share one card that is exactly the wrong
moment to hold it: the agent has just handed a workflow over, and the render
that follows finds 24 of 32 GB taken by a model nobody is asking anything. It
crawls in low-VRAM mode or fails for memory, and the failure reads as a broken
workflow.

So at the handover - just before a graph is queued - :func:`release` asks Ollama
to unload what it holds. The model comes back by itself with the agent's next
call (a load from disk, around half a minute for a 20 GB model), which is the
price of one card doing both jobs.

It does nothing unless a role actually runs on Ollama, and it never raises: a
render must not fail because an unload could not be asked for. Small models are
left where they are - the memory embedder is a few hundred MB and is asked for
all the time.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request

logger = logging.getLogger("agentY.ollama")

# Below this a loaded model is not worth a reload (the embedder is ~0.3 GB).
MIN_BYTES = 2 * 1024 ** 3
TIMEOUT = 4.0
# Several graphs queued in one go ask once, not once each.
_QUIET_SECONDS = 3.0
_last = 0.0


def _settings() -> dict:
    from src.utils.settings import load_settings
    return load_settings()


def wanted(settings: dict | None = None) -> bool:
    """A role runs on Ollama, and the switch (``llm.ollama.unload_before_render``,
    default on) has not been turned off."""
    try:
        cfg = settings if settings is not None else _settings()
        llm = cfg.get("llm") or {}
        if not (llm.get("ollama") or {}).get("unload_before_render", True):
            return False
        specs = list((llm.get("tiers") or {}).values()) + list((llm.get("pipeline") or {}).values())
        return any(str(s or "").strip().lower().startswith("ollama,") for s in specs)
    except Exception:  # noqa: BLE001
        return False


def _call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "{}")


def release(host: str | None = None, *, settings: dict | None = None) -> list[str]:
    """Ask Ollama to unload its large models now. Returns the names it was asked
    to drop (empty when there was nothing to do, or it could not be reached).

    An unload asked for while the model is answering takes effect when that
    answer is done - another conversation mid-call is not cut off.
    """
    global _last
    if not wanted(settings):
        return []
    now = time.monotonic()
    if now - _last < _QUIET_SECONDS:
        return []
    _last = now
    try:
        if host is None:
            from src.utils.settings import ollama_host
            host = ollama_host()
        host = str(host).rstrip("/")
        loaded = _call(host + "/api/ps").get("models") or []
    except Exception as exc:  # noqa: BLE001
        logger.info("ollama: could not ask what is loaded (%s)", exc)
        return []
    dropped = []
    for model in loaded:
        name = str(model.get("name") or model.get("model") or "")
        size = int(model.get("size_vram") or model.get("size") or 0)
        if not name or size < MIN_BYTES:
            continue
        try:
            _call(host + "/api/generate", {"model": name, "keep_alive": 0})
            dropped.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.info("ollama: could not unload %s (%s)", name, exc)
    if dropped:
        logger.info("ollama: unloaded %s before the render", ", ".join(dropped))
    return dropped
