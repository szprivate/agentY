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


def _validation_errors(workflow_path: str) -> list:
    """What the validator says is wrong with the file - [] when it is fine, or
    when it could not be asked (a hand-off is never refused over that)."""
    # Local checks only. ComfyUI's own check queues the graph, and one that is not
    # the only job running is not stopped again - it renders (a Flux inpaint did,
    # for two and a half minutes, at every hand-off of a benchmark run).
    token = None
    try:
        from agenty_core.tools.comfyui import local_only
        token = local_only().set(True)
    except Exception:  # noqa: BLE001 - an older core: no gate rather than a render
        return []
    try:
        from src.tools import validate_workflow as _vw
        got = json.loads(getattr(_vw, "func", _vw)(workflow_path))
    except Exception:  # noqa: BLE001
        return []
    finally:
        try:
            local_only().reset(token)
        except Exception:  # noqa: BLE001
            pass
    if not isinstance(got, dict) or got.get("valid"):
        return []
    errors = [str(e) for e in (got.get("local_errors") or []) if e]
    for node_id, err in ((got.get("server_errors") or {}).get("node_errors") or {}).items():
        for item in (err or {}).get("errors") or []:
            errors.append(f"Node {node_id} ({(err or {}).get('class_type', '?')}): "
                          f"{item.get('message', '')} {item.get('details', '')}".strip())
    return errors


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

    Answers ``status: "not_ready"`` with ``problems`` when the workflow does not
    validate (a required input nothing is wired into, an unknown node). Fix each
    one with ``update_workflow`` and call again.

    Args:
        workflow_path: File path to the validated workflow JSON
                       (the same path returned by ``get_workflow_template`` or
                       ``save_workflow`` and used in ``update_workflow``).
    """
    from src.utils.workflow_signal import (MAX_INVALID_REFUSALS, append_workflow_path,
                                           dead_nodes_refused_once, execution_hold,
                                           invalid_refusals)

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
    # The validator ran after every edit and said what was wrong; nothing made
    # that count. Here it does: a graph with a required input nobody feeds is
    # handed back while the agent still has the turn - twice at most, then it goes
    # through with a note (the run fails and the repair that follows takes it).
    problems = _validation_errors(resolved)
    if problems:
        if invalid_refusals(resolved) <= MAX_INVALID_REFUSALS:
            return json.dumps({
                "status": "not_ready",
                "workflow_path": resolved,
                "problems": problems[:12],
                "problem": f"This workflow does not validate: {len(problems)} problem(s). "
                           "It has NOT been queued.",
                "fix": ("Fix every entry in `problems` with update_workflow - an "
                        "'unconnected' input needs a wire from a node that outputs that "
                        "type (add the node if the graph has none) - until update_workflow "
                        "answers status \"ok\". Then call signal_workflow_ready again."),
            })
        _note(f"⚠️ Running a workflow that still does not validate: {problems[0][:160]}")

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
            "The pipeline will submit it to ComfyUI when your turn ends, run Vision "
            "QA, and save outputs to ./output automatically. Until then it is NOT in "
            "ComfyUI's queue; if you decide it must not run after all, call "
            "withdraw_workflow — an empty ComfyUI queue does not mean it won't run. "
            "For batch runs, call signal_workflow_ready for each duplicate workflow; "
            "otherwise your work here is done — no further tool calls are needed."
        ),
    })


@tool
def withdraw_workflow(workflow_path: str = "") -> str:
    """Take a workflow you signalled this turn back, so it does NOT run.

    ``signal_workflow_ready`` holds a workflow until your turn ends and only then
    submits it to ComfyUI — so ComfyUI's queue shows nothing in the meantime, and
    "nothing is queued" is not true. Call this the moment you decide a signalled
    workflow must not run (a model is missing, the user said not to run it, you
    signalled the wrong file).

    Args:
        workflow_path: The path you signalled. Blank withdraws every workflow
            signalled this turn.
    """
    from src.utils.workflow_signal import peek, withdraw
    gone = withdraw(workflow_path)
    left = peek()
    if gone:
        _note("↩️ Withdrawn — will not run: " + ", ".join(Path(g).name for g in gone))
    return json.dumps({
        "ok": bool(gone),
        "withdrawn": gone,
        "still_signalled": left,
        **({} if gone else {"note": "Nothing matching was signalled this turn."}),
    })


def queue(action: str = "status") -> str:
    """Get or manage the ComfyUI execution queue.

    The answer also lists ``signalled_this_turn``: workflows you passed to
    ``signal_workflow_ready`` that are submitted only when your turn ends, so
    they are not in ComfyUI's queue yet. Use ``withdraw_workflow`` to stop one.

    Args:
        action: 'status' (view queue), 'clear' (clear pending), or 'clear_running' (stop running items).
    """
    from agenty_core.tools.comfyui import queue as _core_queue
    out = _core_queue(action)
    try:
        from src.utils.workflow_signal import peek
        pending = peek()
    except Exception:  # noqa: BLE001
        return out
    if not pending:
        return out
    try:
        data = json.loads(out)
    except Exception:  # noqa: BLE001
        data = {"comfyui": out}
    if not isinstance(data, dict):
        data = {"comfyui": data}
    data["signalled_this_turn"] = pending
    data["note"] = (f"{len(pending)} workflow(s) you signalled will be submitted to ComfyUI "
                    "when this turn ends — withdraw_workflow stops that.")
    return json.dumps(data)


queue = tool(queue)
