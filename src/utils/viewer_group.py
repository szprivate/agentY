"""Put everything the agent saves into one viewer tab, grouped by stage.

The bEpic viewer (ComfyUI-ImageViewer, installed with agentY) gives each
``bEpicSendToViewer`` node a tab of its own unless it is told a ``tab_name``. An
agent run is many such nodes, one per workflow, so its pictures used to be
scattered over tabs named after node ids. Here the node is told, before the
workflow is submitted:

* ``tab_name`` - one tab for every agent image (Settings ▸ ``media_nodes.viewer_tab``,
  "agentY" unless changed; empty leaves the node as it is);
* ``group`` - the stage the picture belongs to: the hook's own title
  ("Characters", "Places"), else the role the user wrote in its prompt;
* ``label`` - what this picture is within the stage, when a batch named its members
  (one character each), so the viewer can say which is which.

The names come from :mod:`src.utils.output_tags`, which the pipeline fills before it
submits anything - the same record that titles the loader nodes dropped on the
canvas and goes into each file's ``.agenty.json`` sidecar. A tab name the user typed
into the node themselves is theirs and is left alone, and so is a viewer too old to
have the ``group`` input.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("agentY.viewer_group")

VIEWER_NODE = "bEpicSendToViewer"
DEFAULT_TAB = "agentY"
_MAX = 60


def tab_name() -> str:
    """The one tab the agent's images go to; "" turns the grouping off."""
    try:
        from src.agent import _load_settings
        chosen = ((_load_settings() or {}).get("media_nodes") or {}).get("viewer_tab")
    except Exception:  # noqa: BLE001
        return DEFAULT_TAB
    return DEFAULT_TAB if chosen is None else " ".join(str(chosen).split())[:_MAX]


def _clean(text) -> str:
    return " ".join(str(text or "").split())[:_MAX]


def names_for(workflow_path) -> tuple[str, str]:
    """``(group, label)`` for the workflow about to run."""
    from src.utils.output_tags import run_role_of
    role, meta = run_role_of(workflow_path)
    group = _clean(meta.get("stage"))
    label = _clean(meta.get("asset") or role)
    if label.lower() == group.lower():
        label = ""
    return group, label


def stamp(graph: dict, workflow_path="", schema_of=None) -> int:
    """Name the viewer nodes of *graph* in place. Returns how many were changed."""
    tab = tab_name()
    if not tab or not isinstance(graph, dict):
        return 0
    nodes = [n for n in graph.values()
             if isinstance(n, dict) and n.get("class_type") == VIEWER_NODE]
    if not nodes:
        return 0
    if schema_of is None:
        from src.utils.media_nodes import _schema as schema_of
    try:
        known = set(((schema_of(VIEWER_NODE) or {}).get("input") or {}).get("optional") or {})
    except Exception:  # noqa: BLE001
        known = set()
    if "group" not in known:
        return 0                       # a viewer from before the grouping: leave it be
    group, label = names_for(workflow_path)
    changed = 0
    for node in nodes:
        inputs = node.setdefault("inputs", {})
        if str(inputs.get("tab_name") or "").strip() not in ("", tab):
            continue                   # the user named this tab themselves
        want = {"tab_name": tab, "group": group, "label": label}
        if any(inputs.get(k) != v for k, v in want.items()):
            inputs.update(want)
            changed += 1
    return changed
