"""Record the dependencies of files made before dependency tracking existed.

Every file agentY or the viewer's Send node writes now records what it was made from
(``agenty_core.utils.dependencies``), and each input lists what was made from it. Older
files have neither, so the viewer's Dependencies tab showed nothing for them. Once the
host is up, this walks ComfyUI's output and input folders in the background, slowly, and
records every file whose graph is stored with it (a sidecar) or embedded in it (a PNG's
or video's ComfyUI metadata). It only adds small JSON files to ``agent_metadata`` folders.

Each folder is walked in full once; later starts look only at files changed since
(``memory/deps_backfill.json``). ``[dependencies] backfill = false`` (or
``AGENTY_DEPS_BACKFILL=0``) switches it off.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("agentY.deps_backfill")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = _PROJECT_ROOT / "memory" / "deps_backfill.json"
START_DELAY_S = 60.0          # let ComfyUI and the first turn come up first
ROOTS_WAIT_S = 600.0          # how long to wait for ComfyUI to say where its folders are

_started = False
_stop = threading.Event()


def enabled() -> bool:
    raw = os.environ.get("AGENTY_DEPS_BACKFILL", "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    try:
        from src.utils.settings import load_settings
        return bool((load_settings().get("dependencies") or {}).get("backfill", True))
    except Exception:  # noqa: BLE001
        return True


def run(delay: float = START_DELAY_S) -> dict:
    """The backfill itself (blocking): wait, find ComfyUI's folders, walk them."""
    from agenty_core.utils import dependencies as dp
    if _stop.wait(delay):
        return {}
    deadline = time.monotonic() + ROOTS_WAIT_S
    rts: dict = {}
    while not _stop.is_set():
        rts = dp.roots()
        if rts.get("output") or time.monotonic() > deadline:
            break
        _stop.wait(30)
    folders = [rts.get("output"), rts.get("input")]
    if not any(folders):
        logger.info("dependencies backfill: ComfyUI's folders are not known; skipped")
        return {}
    got = dp.backfill(folders, STATE_FILE, stop=_stop.is_set, log=logger.info)
    logger.info("dependencies backfill finished: %s looked at, %s recorded", got["looked"], got["recorded"])
    return got


def start() -> bool:
    """Start the backfill thread once per process (no-op when switched off)."""
    global _started
    if _started or not enabled():
        return False
    _started = True
    threading.Thread(target=run, name="agentY-deps-backfill", daemon=True).start()
    return True


def stop() -> None:
    _stop.set()
