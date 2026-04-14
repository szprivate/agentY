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
    """Unload all configured Ollama models from VRAM.

    Sends ``POST /api/generate`` with ``keep_alive=0`` to the Ollama server for
    every model referenced in ``config/settings.json`` (vision + text).  This is
    the documented Ollama mechanism for evicting a model from GPU memory.

    Returns
    -------
    list[str]
        Names of models that were successfully unloaded.  Empty list if Ollama
        is unreachable or no models are configured.
    """
    import httpx
    from src.utils.llm_functions import LLMFunctions

    llm_vis = LLMFunctions.for_vision()
    llm_txt = LLMFunctions.from_settings()
    host = llm_vis.host.rstrip("/")

    # Deduplicate – vision and text model might be the same tag.
    models_to_unload = list(dict.fromkeys([llm_vis.model, llm_txt.model]))
    unloaded: list[str] = []

    for model in models_to_unload:
        try:
            resp = httpx.post(
                f"{host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=10.0,
            )
            resp.raise_for_status()
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

