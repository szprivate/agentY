"""Tool that restarts the agent process."""

import os
import sys

from strands import tool


@tool
def restart_agent() -> str:
    """Restart the agent process.

    Replaces the current process with a fresh instance of the same Python
    interpreter and command-line arguments.  Any external process supervisor
    (e.g. systemd, PM2, Docker restart policy) will then bring the agent
    back up automatically.

    This tool is intended to be called during triage when the user explicitly
    asks to restart or reset the agent.
    """
    os.execv(sys.executable, [sys.executable] + sys.argv)
    # os.execv never returns on success; the line below is unreachable.
    return "Restarting…"  # pragma: no cover
