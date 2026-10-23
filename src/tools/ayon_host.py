"""What agentY lends the Ayon tools (agenty_core.tools.ayon): the canvas, and a way to edit it in place.

The core tools work without a host (they read the nodes the user has selected, and hand back node data to
paste). In agentY they get more:

* the **canvas provider**: the API-format graph the panel posted with this turn's message, so the AYON Context
  node and the node ids are known without the user selecting anything;
* the **canvas placer**: adds / wires nodes through the panel's ``edit_graph`` patch, the same path
  ``edit_canvas_graph`` uses. ``/agent/load_workflow`` would reload the tab under another name, and Ayon finds
  its workfile by the tab name.
"""

from __future__ import annotations

from typing import Any

from agenty_core.tools import ayon


def _turn_graph() -> dict | None:
    from agenty_core.utils import turn_scope
    graph = turn_scope.current().get("canvas_prompt")
    return graph if isinstance(graph, dict) and graph else None


def canvas_provider() -> dict | None:
    """This turn's canvas; failing that, the freshest snapshot the panel has posted (Slack turns, MCP callers)."""
    graph = _turn_graph()
    if graph:
        return graph
    try:
        from src.utils.agentY_server import request_canvas
        snap = request_canvas(wait=0.0) or {}
        prompt = snap.get("prompt")
        return prompt if isinstance(prompt, dict) and prompt else None
    except Exception:  # noqa: BLE001
        return None


def canvas_placer(ops: list[dict], reason: str) -> bool:
    """Push the ops to the panel as one undoable edit, and mirror them into this turn's graph."""
    from src.utils.canvas_patch import push
    if not ops:
        return False
    push({"op": "edit_graph", "ops": ops, "reason": reason})
    graph = _turn_graph()
    if graph is not None:           # keep the turn's graph the one the other canvas tools see
        for op in ops:
            if op.get("op") == "add":
                graph[str(op["node_id"])] = {"class_type": op["class_type"],
                                             "inputs": dict(op.get("params") or {})}
            elif op.get("op") == "connect" and str(op.get("to")) in graph:
                graph[str(op["to"])]["inputs"][op["input"]] = [str(op["from"]), int(op.get("output") or 0)]
    return True


def install() -> None:
    ayon.set_canvas_provider(canvas_provider)
    ayon.set_canvas_placer(canvas_placer)
