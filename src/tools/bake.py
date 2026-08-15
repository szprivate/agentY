"""``bake_hooks_to_canvas`` — bake a chain of generated stage-workflows into a
ComfyUI subgraph canvas (the hook's ``bake`` switch, on a make_workflow hook).

The orchestrator generates one workflow per make_workflow stage as usual; this tool then
nests each into a ComfyUI subgraph whose exposed inputs/outputs match that stage's
hook slots, places one subgraph instance per stage on a single canvas, wires them
to mirror the hook chain, and pushes the result onto the ComfyUI canvas. The baked
workflow is a self-contained native graph the user can re-run without the agent.

Heavy lifting (subgraph JSON assembly) is in ``src.utils.subgraph_bake``; here we
load each stage's workflow, convert API→graph, and push the result.
"""
from __future__ import annotations

import json
from pathlib import Path

from strands import tool


def _to_graph(workflow: dict) -> dict:
    """Convert an API/prompt-format workflow to graph format (v0.4); pass a graph
    through unchanged. Reuses agenty_core's converters."""
    from agenty_core.tools.comfyui import _api_to_graph, _is_graph_format
    return workflow if _is_graph_format(workflow) else _api_to_graph(workflow)


@tool
def bake_hooks_to_canvas(stages: list, links: list | None = None) -> str:
    """Bake generated make_workflow stage workflows into a chained ComfyUI subgraph canvas.

    Call this for make_workflow hooks whose ``bake`` switch is ON, AFTER you have
    generated (and validated) each stage's workflow. Each stage's workflow is
    nested into a ComfyUI subgraph whose inputs/outputs match that hook's slots;
    the subgraph instances are placed on one canvas and wired to mirror the hook
    chain, then pushed onto the ComfyUI canvas — "baking" the multi-step task into
    a reusable native workflow.

    Slots are type-agnostic, so exported outputs may be images, video, OR scalars
    (STRING / INT / FLOAT).

    Args:
        stages: Ordered list (subgraph order) of stage dicts. Each stage:
            - ``workflow_path`` (str): path to that stage's generated workflow
              (API or graph format).
            - ``name`` (str): a SHORT name for the subgraph — 2 to 5 words saying
              what the stage DOES, e.g. "Upscale 2x + grain", "Animate the scene",
              "Extract last frame". Do NOT pass the directive: it is a paragraph,
              and this name is rendered on a collapsed node, in the breadcrumb and
              in the node library. Anything longer is shortened for you, so a bad
              name is a worse name rather than a broken canvas.
            - ``inputs`` (list): inner inputs to expose, in the order matching the
              hook's input slots — ``[{"node_id": <id>, "input": <input name>,
              "type": <optional slot type>}]``. These are the inner-node inputs the
              incoming connection should feed.
            - ``outputs`` (list): inner outputs to expose, matching the hook's
              output slots — ``[{"node_id": <id>, "output_slot": <int>,
              "type": <optional slot type>}]``.
            - ``computed_outputs`` (list, optional): values you computed OUTSIDE the
              graph at runtime (e.g. a video's length) that should still become
              native subgraph outputs. Each is
              ``[{"code": <python snippet>, "inputs": [{"node_id": <id>,
              "output_slot": <int>}], "type": <optional>, "name": <optional>}]``. The
              bake step injects an AgentYPython node running your snippet — the same
              one you used at runtime — with the referenced inner outputs bound as
              ``in0, in1, …``; the snippet must set a list ``outputs`` whose
              ``outputs[0]`` becomes the exposed subgraph output. This makes the
              value reproducible on re-run without the agent.
        links: Optional explicit wiring between stages
            ``[{"from_stage": <idx>, "from_output": <int>, "to_stage": <idx>,
            "to_input": <int>}]`` (indices into ``stages``). Omit for a linear
            chain (each stage's output 0 → the next stage's input 0).

    Returns:
        A JSON status string ``{status, subgraphs, saved_as?, opened_on_canvas?}``.
    """
    from src.utils.subgraph_bake import build_baked_workflow, short_name as _short

    if not isinstance(stages, list) or not stages:
        return json.dumps({"status": "error", "error": "stages must be a non-empty list"})

    built_stages: list[dict] = []
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            return json.dumps({"status": "error", "error": f"stage {i} is not an object"})
        wf_path = st.get("workflow_path")
        if not wf_path or not Path(wf_path).exists():
            return json.dumps({"status": "error",
                               "error": f"stage {i}: workflow_path not found: {wf_path!r}"})
        try:
            workflow = json.loads(Path(wf_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return json.dumps({"status": "error", "error": f"stage {i}: cannot read workflow: {e}"})
        try:
            graph = _to_graph(workflow)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"stage {i}: API→graph failed: {e}"})
        built_stages.append({
            "graph": graph,
            # Shortened rather than trusted — see subgraph_bake.short_name. A
            # directive pasted in whole is the failure that shows up on the canvas,
            # and it shows up on every stage at once.
            "name": _short(st.get("name"), fallback=f"Stage {i + 1}"),
            "inputs": st.get("inputs") or [],
            "outputs": st.get("outputs") or [],
            "computed_outputs": st.get("computed_outputs") or [],
        })

    try:
        baked = build_baked_workflow(built_stages, links=links)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "error": f"subgraph assembly failed: {e}"})

    # Mark the payload additive so the canvas MERGES the baked subgraphs into the
    # current graph (keeping the user's hook nodes and everything else) instead of
    # replacing it — see web/agent_canvas.js (graph.extra.agentY_add).
    baked["extra"] = {**(baked.get("extra") or {}), "agentY_add": True}

    try:
        from agenty_core.tools.comfyui import get_client
        resp = get_client().post("/agent/load_workflow", json_data={"workflow": baked})
        opened = isinstance(resp, dict) and bool(resp.get("ok"))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "partial",
                           "error": f"built {len(built_stages)} subgraph(s) but push failed: {e}",
                           "subgraphs": len(built_stages)})

    return json.dumps({
        "status": "ok",
        "subgraphs": len(built_stages),
        "opened_on_canvas": opened,
        "message": (f"Baked {len(built_stages)} stage(s) into chained subgraphs and ADDED them "
                    "to the canvas alongside the hook nodes (nothing removed). The user can "
                    "re-run this native workflow without the agent."),
    })
