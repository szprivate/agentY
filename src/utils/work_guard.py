"""A turn that ends with the graph it worked on still broken.

The validator runs after every edit and says what is wrong - and that was all it
did. Measured with a small local orchestrator (qwen3.5:9b, benchmark of
2026-10-13): ``update_workflow`` answered "error" seven times in a row and the
turn then simply ended; ``edit_canvas_graph`` put a VAEEncode on the canvas with
nothing wired into it, said so, and the turn ended. A model that reads a result
and carries on needs nothing more. One that does not is left with a graph that
cannot run and an answer that says it was built.

So the harness looks once more when the turn ends, at what the turn itself left
behind:

  * the open graph: required inputs of the nodes this turn added or rewired
    that nothing feeds (the pipeline keeps that list, :func:`graph_note`);
  * a workflow file: the last ``update_workflow`` of the turn answered "error"
    and nothing was handed off (:func:`last_file_edit`).

and sends the turn back with the exact list - a file after the repair
specialist has had a go at it. Bounded (:data:`MAX_REDRIVES`), and never for a
turn that changed nothing.
"""

from __future__ import annotations

import json

MAX_REDRIVES = 2
_EDIT_TOOLS = ("update_workflow",)


def _result_text(block: dict) -> str:
    parts = []
    for item in (block.get("toolResult") or {}).get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def last_file_edit(messages, start: int = 0) -> dict | None:
    """The LAST ``update_workflow`` result among *messages* from *start* on, as
    ``{"ok": bool, "workflow_path": str, "errors": [...]}`` - or None when the
    turn made no such edit."""
    names: dict[str, str] = {}
    last = None
    for msg in list(messages or [])[max(0, int(start or 0)):]:
        if not isinstance(msg, dict):
            continue
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            use = block.get("toolUse")
            if isinstance(use, dict):
                names[str(use.get("toolUseId"))] = str(use.get("name") or "")
                continue
            res = block.get("toolResult")
            if not isinstance(res, dict) or names.get(str(res.get("toolUseId"))) not in _EDIT_TOOLS:
                continue
            try:
                data = json.loads(_result_text(block))
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict) or "status" not in data:
                continue
            errors = []
            for key in ("node_errors", "local_errors", "problems"):
                errors += [str(e) for e in (data.get(key) or []) if e]
            server = data.get("server_errors")
            if server:
                errors.append("ComfyUI: " + json.dumps(server)[:400])
            last = {"ok": str(data.get("status")) == "ok", "workflow_path": str(data.get("workflow_path") or ""),
                    "errors": errors}
    return last


def turn_targets(messages, start: int = 0) -> set[str]:
    """Workflow files this turn itself produced: a ``workflow_path`` in the result
    of any tool other than the edit tools (prepare_workflow, generate_new_workflow,
    duplicate_workflow, fix_workflow_assembly ...). A file the turn merely patched
    - an older one from a previous turn - is not the turn's target, and a failed
    edit of it must not set off a repair, a hand-off or a run nobody asked for."""
    names: dict[str, str] = {}
    out: set[str] = set()
    for msg in list(messages or [])[max(0, int(start or 0)):]:
        if not isinstance(msg, dict):
            continue
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            use = block.get("toolUse")
            if isinstance(use, dict):
                names[str(use.get("toolUseId"))] = str(use.get("name") or "")
                continue
            res = block.get("toolResult")
            if not isinstance(res, dict) or names.get(str(res.get("toolUseId"))) in _EDIT_TOOLS:
                continue
            try:
                data = json.loads(_result_text(block))
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and data.get("workflow_path"):
                out.add(str(data["workflow_path"]))
    return out


def graph_note(unwired: dict, hints: dict | None = None) -> str:
    """The list for the note: one line per node, each unfed input with where it
    could be wired from."""
    lines = []
    for node_id, inputs in (unwired or {}).items():
        for name in inputs or []:
            line = f"- node {node_id}: `{name}` has nothing wired into it"
            got = ((hints or {}).get(str(node_id)) or {}).get(name)
            if got:
                line += " - could come from " + "; ".join(got)
            lines.append(line)
    return "\n".join(lines)
