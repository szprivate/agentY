"""Is this host a server with no ComfyUI canvas behind it?

``server.headless`` (or ``AGENTY_HEADLESS=1``) says nobody is expected to have the
ComfyUI tab open. Three things follow, and this module is the one place that
answers the question so they cannot disagree:

* finished media is not dropped onto a canvas as loader nodes - Slack (and the
  output folder) is where results go;
* the agent does not wait for a canvas snapshot that cannot come;
* the tools that need the page (screenshots, open tabs, hook runs on the live
  graph) answer at once, in words that say why, instead of timing out.

Off by default; a desktop install never sees any of it.
"""
from __future__ import annotations

import os


def headless() -> bool:
    env = str(os.environ.get("AGENTY_HEADLESS", "")).strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from src.utils.settings import load_settings
        return bool((load_settings().get("server") or {}).get("headless", False))
    except Exception:  # noqa: BLE001 - unreadable settings mean "not headless"
        return False


NO_CANVAS = (
    "this agentY host runs headless (server.headless): there is no ComfyUI canvas "
    "or browser tab behind it, so this needs a canvas it cannot have. Build and run "
    "the workflow directly (execute_workflow); results are delivered to Slack and "
    "saved in the output folder."
)


def no_canvas_error(what: str = "") -> dict:
    """The reply for a canvas-only tool on a headless host."""
    out = {"error": NO_CANVAS, "headless": True}
    if what:
        out["tool"] = what
    return out
