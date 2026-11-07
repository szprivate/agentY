"""``file_dependencies`` - what a file was made from, what was made from it, and what is out of date.

The record is kept with the files (``agenty_core.utils.dependencies``): every
output's sidecar names the files its graph loaded, and every input lists what was
made from it. A pass that already exists (a mask, depth, pose of the same frames)
can be reused instead of made again, and a plate that changed shows which passes
and renders were made from the old one.
"""
from __future__ import annotations

import json
import os

from strands import tool

_MAX_ITEMS = 40


def _brief_params(params: dict) -> dict:
    return {k: v for k, v in (params or {}).items() if not isinstance(v, str) or len(v) <= 120}


@tool
def file_dependencies(path: str, scan_folder: bool = False) -> str:
    """What a file was made from, what has been made from it, and whether it is out of date.

    Use it before making a pass of a clip or image (mask, depth, pose, edges, camera
    solve, key): if one with the same frames and settings is listed under
    ``made_from_this``, reuse it. Use it when a plate or source changed ("I re-graded
    the plate", "the edit changed") to list what was made from the old one, and when
    the user asks what a file came from or what depends on it.

    Args:
        path: The file (absolute path, or a name in ComfyUI's input/output folder).
        scan_folder: Also record the files in its folder made before this record
            existed (slower; use it when ``made_from_this`` looks incomplete).

    Returns JSON ``{status, file, kind, stale, made_from, made_from_this}``:
    ``made_from`` items are ``{file, path, changed, exists}``; ``made_from_this``
    items are ``{file, path, kind, params, workflow, created, exists, stale}``.
    ``stale`` is why the file is out of date, or "".
    """
    from agenty_core.utils import dependencies as dp
    p = path
    if p and not os.path.isfile(p):
        p = dp.resolve_value(p) or p
    if not p or not os.path.isfile(p):
        return json.dumps({"status": "error", "error": f"no such file: {path}"})
    if scan_folder:
        try:
            dp.reindex(os.path.dirname(os.path.abspath(p)))
        except Exception:  # noqa: BLE001
            pass
    deps = dp.deps_of(p)
    memo: dict = {}
    made_from = [{"file": os.path.basename(i["path"]), "path": i["path"],
                  "changed": i["changed"], "exists": i["exists"]} for i in dp.inputs_of(p)]
    made = []
    for i in reversed(dp.derived_of(p)):
        if len(made) >= _MAX_ITEMS:
            break
        made.append({"file": os.path.basename(i["path"]), "path": i["path"], "kind": i.get("kind", ""),
                     "params": _brief_params(i.get("params") or {}), "workflow": i.get("workflow", ""),
                     "created": i.get("created", ""), "exists": i.get("exists", False),
                     "stale": dp.stale_reason(i["path"], memo) if i.get("exists") else ""})
    return json.dumps({"status": "ok", "file": os.path.abspath(p), "kind": deps.get("kind", ""),
                       "stale": dp.stale_reason(p, memo), "made_from": made_from,
                       "made_from_this": made}, ensure_ascii=False, default=str)
