"""Record a successfully run workflow as a machine-local template (the fast path for solved jobs).

The store, the matching and the folder live in :mod:`agenty_core.tools.local_templates`;
this is the host's half: WHEN a run is worth keeping.

Conservative on purpose - a wrong saved job costs more than the research it saves:

* only after a run that finished with outputs, with no failed member and no QA
  failure (an ordinary turn's runs are recorded by ``Pipeline`` after its batch;
  ``partner_run`` records its own);
* never a dry run, a hook chain's stage, or the user's own canvas graph;
* the request must be a real sentence (the words the job is matched on);
* an API graph of a sensible size with at least one working node;
* the same structure again is ONE template whose use count goes up;
* these are stored as ``source: "auto"``, which matches only on stronger wording
  than a job the user asked to keep (``register_workflow_template(local=True)``);
* switch off with the setting ``local_templates_auto = false``.

Forget one with ``unregister_workflow_template(name)``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("agentY.local_templates")

_PLUMBING = ("Load", "Save", "Preview", "Primitive", "Reroute", "Note", "MarkdownNote")
MAX_NODES = 80
MIN_REQUEST_CHARS = 15


def enabled() -> bool:
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("local_templates_auto", True))
    except Exception:  # noqa: BLE001
        return True


def eligible(graph, request: str, outputs) -> str:
    """``""`` when the run is worth keeping, else the reason it is not."""
    if len(" ".join(str(request or "").split())) < MIN_REQUEST_CHARS:
        return "the request is too short to match on"
    if not outputs:
        return "it produced no output"
    if not isinstance(graph, dict) or not graph:
        return "not an API graph"
    nodes = {k: v for k, v in graph.items() if isinstance(v, dict) and v.get("class_type")}
    if not nodes or len(nodes) != len(graph):
        return "not an API graph"
    if len(nodes) > MAX_NODES:
        return f"{len(nodes)} nodes is more than a job template should carry"
    classes = [str(n["class_type"]) for n in nodes.values()]
    if any("agentY" in c for c in classes):
        return "it carries agentY hook nodes"
    if all(c.startswith(_PLUMBING) for c in classes):
        return "nothing in it does any work"
    return ""


def record(workflow_path: str, request: str, outputs, source: str = "auto") -> str:
    """Save the workflow at *workflow_path* when it qualifies. Returns the template name, or ``""``."""
    if not enabled():
        return ""
    try:
        graph = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if isinstance(graph.get("prompt"), dict):
        graph = graph["prompt"]
    why = eligible(graph, request, outputs)
    if why:
        logger.debug("local template: %s not kept - %s", workflow_path, why)
        return ""
    try:
        from agenty_core.tools import local_templates
        got = local_templates.save_local(graph, request=request, source=source, origin=str(workflow_path))
        logger.info("local template %s %s", "saved" if got["created"] else "refreshed", got["name"])
        return got["name"]
    except Exception as exc:  # noqa: BLE001 - never cost a run its report
        logger.warning("local template not saved: %s", exc)
        return ""


def record_turn(paths, request: str, outputs) -> list:
    """Record each workflow a turn ran (one call per turn, after its batch)."""
    names = []
    for p in dict.fromkeys(str(x) for x in (paths or []) if x):
        got = record(p, request, outputs)
        if got:
            names.append(got)
    return names
