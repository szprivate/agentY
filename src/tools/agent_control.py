"""Agent process control utilities for agentY."""

from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger("agentY.agent_control")

# User messages that trigger a full process restart (lowercased, stripped).
RESTART_COMMANDS: frozenset[str] = frozenset({"restart", "restart agent", "!restart"})


def is_restart_command(text: str) -> bool:
    """Return True if *text* is a restart command."""
    return text.strip().lower() in RESTART_COMMANDS


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

