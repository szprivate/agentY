"""Agent process control utilities for agentY."""

from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger("agentY.agent_control")

# User messages that trigger a full process restart (lowercased, stripped).
RESTART_COMMANDS: frozenset[str] = frozenset({"restart", "restart agent", "!restart"})

# User messages that trigger an Ollama VRAM unload (lowercased, stripped).
UNLOAD_COMMANDS: frozenset[str] = frozenset({"/unload", "unload", "unload models", "!unload"})


def is_restart_command(text: str) -> bool:
    """Return True if *text* is a restart command."""
    return text.strip().lower() in RESTART_COMMANDS


def is_unload_command(text: str) -> bool:
    """Return True if *text* is an Ollama unload command."""
    return text.strip().lower() in UNLOAD_COMMANDS


def unload_ollama_models() -> list[str]:
    """Unload every Ollama model currently resident in VRAM.

    Queries ``GET /api/ps`` to discover what is actually loaded, then sends
    ``POST /api/generate`` with ``keep_alive=0`` for each one.  This is more
    accurate than unloading the configured models statically — agents may load
    other tags (e.g. via ``OllamaModel``), and previously-configured tags may
    no longer be loaded.  When nothing is loaded the call is a single ``/api/ps``
    HTTP request and returns immediately.

    Returns
    -------
    list[str]
        Names of models that were successfully unloaded.  Empty list if Ollama
        is unreachable or no models were resident.
    """
    import httpx
    from src.utils.llm_functions import LLMFunctions

    host = LLMFunctions.from_settings().host.rstrip("/")

    try:
        resp = httpx.get(f"{host}/api/ps", timeout=5.0)
        resp.raise_for_status()
        loaded_raw = resp.json().get("models", []) or []
    except Exception as exc:
        logger.debug("agent_control: /api/ps unreachable, skipping unload: %s", exc)
        return []

    loaded: list[str] = []
    for entry in loaded_raw:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
            if name and name not in loaded:
                loaded.append(name)

    if not loaded:
        return []

    unloaded: list[str] = []
    for model in loaded:
        try:
            r = httpx.post(
                f"{host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=10.0,
            )
            r.raise_for_status()
            logger.info("agent_control: unloaded Ollama model '%s' from VRAM", model)
            unloaded.append(model)
        except Exception as exc:
            logger.debug("agent_control: could not unload Ollama model '%s': %s", model, exc)

    return unloaded


def restart_process(delay: float = 0.0) -> None:
    """Replace the running process with a fresh copy of itself.

    Uses ``os.execv`` so the new process inherits the same PID slot and
    command-line arguments.  Falls back to Popen + exit on platforms where
    execv is unavailable (e.g. Windows embedded interpreters).

    Args:
        delay: Seconds to wait before replacing the process (allows callers
               to flush output before replacing the process.
    """
    def _do_restart() -> None:
        logger.info("Restarting agent process: %s %s", sys.executable, sys.argv)
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as exc:
            logger.warning("os.execv failed (%s); falling back to Popen + exit", exc)
            import subprocess  # noqa: PLC0415
            subprocess.Popen(  # noqa: S603
                [sys.executable] + sys.argv,
                close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            os._exit(0)

    if delay:
        threading.Timer(delay, _do_restart).start()
    else:
        _do_restart()

