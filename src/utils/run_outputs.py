"""What a run's output files are, for the agent that judges them.

The promo run of 2026-11-15 spent 139 tool calls on one look frame because of
this. The Z-Image graph it built had a ``PreviewImage`` on the scaled INPUT
frame, and ``run_workflow_now`` listed that preview (``ComfyUI_temp_*.png`` in
ComfyUI's temp folder) first. The agent analysed it after every change of
denoise. ComfyUI cached that node, so the file was written once and never again.
Each time the agent saw "grey clay, composition kept" and raised the denoise.
The real results in ``output/agent/`` were good and were never looked at.

So every output is labelled:

* ``saved``: a file in ComfyUI's output folder (a Save node wrote it). It is the
  result and comes first.
* ``preview``: a file in the temp folder (Preview nodes, ``ComfyUI_temp_*``). It is
  not a result. In a graph that previews its input, it is the INPUT.
* ``fresh``: written by THIS run (its mtime is at or after the run's start). A file
  that is not fresh is a cached node's old file.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Slack for clocks and file systems that round mtimes (network shares: 2 s).
_MTIME_SLACK_S = 2.0


def _is_preview(path: str) -> bool:
    p = str(path or "").replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name.startswith("ComfyUI_temp_"):
        return True
    parts = [x.lower() for x in p.split("/")[:-1]]
    return "temp" in parts


def describe(paths: list, started: float | None) -> list[dict]:
    """One entry per path, saved results first and in their own order.

    Each entry is ``{path, file, subfolder, kind, fresh}``. ``file`` and
    ``subfolder`` are what ``view_image`` takes (``subfolder`` is the part between
    the output or temp folder and the file). ``fresh`` is None when the file cannot
    be stat'ed or no start time is known.
    """
    rows = []
    for raw in paths or []:
        path = str(raw)
        kind = "preview" if _is_preview(path) else "saved"
        fresh = None
        if started is not None:
            try:
                fresh = os.path.getmtime(path) >= started - _MTIME_SLACK_S
            except OSError:
                fresh = None
        norm = path.replace("\\", "/")
        parts = norm.split("/")
        sub = ""
        for root in ("output", "temp"):
            low = [x.lower() for x in parts[:-1]]
            if root in low:
                i = len(low) - 1 - low[::-1].index(root)
                sub = "/".join(parts[i + 1:-1])
                break
        rows.append({"path": path, "file": parts[-1], "subfolder": sub,
                     "kind": kind, "fresh": fresh})
    rows.sort(key=lambda r: (r["kind"] != "saved", r["fresh"] is False))
    return rows


def note(rows: list[dict]) -> str:
    """A sentence for the tool's message, or "" when every output is a fresh result."""
    saved = [r for r in rows if r["kind"] == "saved"]
    previews = [r for r in rows if r["kind"] == "preview"]
    stale = [r for r in rows if r["fresh"] is False]
    bits = []
    if saved:
        bits.append("Judge the SAVED result(s): " + ", ".join(r["path"] for r in saved) + ".")
    if previews:
        bits.append("Preview(s) " + ", ".join(Path(r["path"]).name for r in previews)
                    + " are not results: a Preview node wrote them to the temp folder, "
                      "and in a graph that previews its input they show the INPUT.")
    if stale:
        bits.append("Not rewritten by this run (a cached node's old file): "
                    + ", ".join(Path(r["path"]).name for r in stale)
                    + ". Looking at it again tells you nothing about the change you made.")
    if not previews and not stale:
        return ""
    return " ".join(bits)
