"""agentY-only pipeline handoff tool.

``signal_workflow_ready`` is specific to the Strands multi-agent pipeline: the
Assemble Workflow calls it as its final step instead of executing the workflow itself, and
the Executor stage (submission, polling, Ollama Vision-QA, output saving) takes
over.  It is therefore NOT part of the shared ``agenty_core`` tool layer — it
lives here, in the agentY repo only.  The agentY-mcp server has no equivalent
(its host model calls ``execute_workflow`` directly).
"""

import json
from pathlib import Path

from strands import tool


def _note(text: str) -> None:
    """Put a line in the run stream, and never fail for it."""
    try:
        from agenty_core.utils.progress_signal import push
        push(text)
    except Exception:  # noqa: BLE001
        print(text)


def _dead_nodes(workflow_path: str) -> tuple[list, list]:
    """``(dead_nodes, warning lines)`` for the graph about to be queued.

    Cheap on purpose — the graph plus cached ``/object_info``, no server
    validation — because a batch signals this once per member. Any failure means no
    finding: a check on the way to a render must never be the reason there is no
    render.
    """
    try:
        from agenty_core.tools.assembly_deterministic import dead_nodes_in_file
        return dead_nodes_in_file(workflow_path)
    except Exception:  # noqa: BLE001
        return [], []


@tool
def signal_workflow_ready(workflow_path: str) -> str:
    """Signal that the workflow is fully assembled and validated, ready for execution.

    Call this as your **final step** once ``update_workflow()`` returns ``status: "ok"``.
    The pipeline will automatically handle ComfyUI submission, completion
    polling, Vision QA (via Ollama), and saving outputs to ``./output``.

    For **batch runs** (``count_iter > 1``): call this tool once for every
    workflow file produced by ``duplicate_workflow()``.  Each call appends to
    the execution queue; the pipeline submits them to ComfyUI in order.

    Do NOT call ``submit_prompt`` — this tool replaces it.

    Answers ``status: "not_ready"`` with ``dead_nodes`` when the graph carries nodes
    ComfyUI would never execute (their output read by nobody). Fix each one and call
    again — that call runs.

    Args:
        workflow_path: File path to the validated workflow JSON
                       (the same path returned by ``get_workflow_template`` or
                       ``save_workflow`` and used in ``update_workflow``).
    """
    from src.utils.workflow_signal import (append_workflow_path,
                                           dead_nodes_refused_once, execution_hold)

    # A turn whose plan the user asked to approve holds the queue shut until they
    # have answered. Refusing here — while the agent still has the turn — is what
    # lets it present the plan instead of announcing a run that never happened.
    hold = execution_hold()
    if hold:
        try:
            from agenty_core.utils.progress_signal import push as _push
            _push("✋ The plan was asked to be approved first — holding.")
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(hold)

    # Paths come from an LLM, sometimes wrapped in newlines; one made a file that
    # exists report "not found".
    p = Path(str(workflow_path).strip())
    if not p.exists():
        return json.dumps({"error": f"Workflow file not found: {workflow_path}"})

    resolved = str(p.resolve())

    # Last look before anything is paid for. ComfyUI executes a graph backwards
    # from its output nodes, so a node whose output nothing reads is skipped in
    # silence — it fails no validation, local or server-side, and the run reports
    # success while doing less than it says (a decoded audio branch that never
    # reached the video node gave a silent video out of an audio model). The build
    # gate catches this for workflows built from scratch; this catches it for a
    # patched template too, which is where that one came from.
    #
    # Handed back ONCE, while the agent still has the turn to fix it. A second
    # signal of the same file is queued with the note anyway: something useless in
    # a graph must never become a render that never happens.
    dead, lines = _dead_nodes(resolved)
    if dead and dead_nodes_refused_once(resolved):
        return json.dumps({
            "status": "not_ready",
            "workflow_path": resolved,
            "dead_nodes": dead,
            "problem": (f"{len(dead)} node(s) in this workflow will never be executed. "
                        "ComfyUI runs a graph backwards from its output nodes, so a node "
                        "whose output nothing reads is skipped entirely — it does not "
                        "fail validation, it just silently does not happen."),
            "warnings": lines,
            "fix": ("For each one: wire its output into the branch that reaches the "
                    "output node if it was meant to do something — that is the usual "
                    "case, and it means this graph is not doing what was asked (a "
                    "decoded audio output belongs in the video node's `audio` input; a "
                    "size-deriving node's width/height belong in the latent). Otherwise "
                    "remove it with update_workflow(remove_nodes=[...]). Then call "
                    "signal_workflow_ready again — the next call runs, so do not signal "
                    "until you have dealt with each one."),
        })
    if dead:
        _note(f"⚠️ Running anyway with {len(dead)} node(s) that will never execute: "
              + ", ".join(f"{d['node_id']} ({d['class_type']})" for d in dead))

    append_workflow_path(resolved)
    # A dry run still queues here — the path is what the end-of-turn summary
    # reports on — but the submission itself never happens. Saying "ready" would
    # have the agent announce a generation that was deliberately skipped.
    try:
        from src.utils import dry_run as _dry
        if _dry.active():
            return json.dumps({
                "status": "dry_run",
                "workflow_path": resolved,
                "message": (
                    "DRY RUN — the workflow is built and can be opened at the path above, "
                    "and it will NOT be submitted to ComfyUI. Your work here is done; tell "
                    "the user what it would have produced."
                ),
            })
    except Exception:  # noqa: BLE001
        pass
    return json.dumps({
        "status": "ready",
        "workflow_path": resolved,
        "message": (
            "Workflow has been added to the execution queue. "
            "The pipeline will submit it to ComfyUI, run Vision QA, "
            "and save outputs to ./output automatically. "
            "For batch runs, call signal_workflow_ready for each duplicate workflow; "
            "otherwise your work here is done — no further tool calls are needed."
        ),
    })
