"""agentY's ``regenerate_from_source``: recreate shots through the host's own run path.

The tool itself lives in agentY-core (``agenty_core.tools.regenerate``): it reads a
shot's stored graph, applies ONLY the requested changes and saves the result as the
next version beside the original. What a host adds is HOW each shot runs. Here that
is ``run_workflow_now`` - the same path every chained run of this conversation takes,
with its gates (plan approval, review stops, the cost confirmation and the credit
balance), its dry-run mode, QA against the user's briefing, the progress lines in the
panel, and the outputs staged as the turn's own.

Core then moves the produced file to its versioned name; the listener below points the
conversation's lists of this turn's outputs at the new place, so what is staged and
shown at the end of the turn is the new version, not a path that is gone.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading

#: One recreate at a time per process: core's runner is one module-level slot.
_RUN_LOCK = threading.Lock()
#: How long one shot may take through run_workflow_now before the tool gives up on it.
SHOT_TIMEOUT_S = 3600.0


def result_of(raw: str) -> dict:
    """run_workflow_now's answer as core's runner result (``status``, ``outputs``, ``error``,
    ``error_kind``)."""
    try:
        doc = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    except (TypeError, ValueError):
        return {"status": "error", "error": f"unexpected answer: {str(raw)[:200]}", "outputs": [], "prompt_id": ""}
    if not isinstance(doc, dict):
        return {"status": "error", "error": "unexpected answer", "outputs": [], "prompt_id": ""}
    outputs = [str(p) for p in (doc.get("outputs") or []) if p]
    out = {"status": "done" if outputs else "error", "outputs": outputs, "prompt_id": ""}
    if doc.get("error_kind"):
        out["error_kind"] = str(doc["error_kind"])
    if doc.get("status") == "cached":
        # Identical inputs: ComfyUI handed back the earlier file - not a new version.
        return {"status": "error", "outputs": [], "prompt_id": "",
                "error": "ComfyUI served the earlier result from its cache - nothing changed"}
    if doc.get("dry_run") or doc.get("status") == "dry_run":
        return {"status": "error", "outputs": [], "prompt_id": "",
                "error": "this conversation is in dry-run mode: nothing was rendered"}
    if not outputs:
        out["error"] = str(doc.get("error") or doc.get("reason") or doc.get("message")
                           or "the run produced no output")[:400]
    # Which run and node made each file: core reads that from the provenance the
    # executor wrote beside the outputs (regenerate._from_recorded).
    return out


def _repoint(lists, src: str, dst: str) -> None:
    """Replace *src* by *dst* in each of the conversation's output lists (in place)."""
    keys = {os.path.normcase(os.path.abspath(src))}
    for lst in lists:
        if not isinstance(lst, list):
            continue
        for i, p in enumerate(lst):
            try:
                if os.path.normcase(os.path.abspath(str(p))) in keys:
                    lst[i] = dst
            except (TypeError, ValueError):
                continue


def make_tool(pipeline, run_workflow_now, tool_decorator):
    """The ``regenerate_from_source`` tool for *pipeline*: core's tool, run through
    *run_workflow_now* (the pipeline's closure)."""
    from agenty_core.tools import regenerate as rg

    async def regenerate_from_source(paths: list, changes: list | None = None, new_seed: bool = False,
                                     dry_run: bool = True, confirm_cost: bool = False) -> str:
        core = getattr(rg.regenerate_from_source, "_tool_func", rg.regenerate_from_source)
        if dry_run:
            return await asyncio.to_thread(core, paths, changes, new_seed, True, confirm_cost)
        loop = asyncio.get_running_loop()

        def runner(workflow_path, *, confirm_cost=False):
            fut = asyncio.run_coroutine_threadsafe(
                run_workflow_now(workflow_path=str(workflow_path), confirm_cost=bool(confirm_cost)), loop)
            try:
                raw = fut.result(SHOT_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                fut.cancel()
                return {"status": "error", "error": f"run failed: {str(exc)[:200]}", "outputs": [], "prompt_id": ""}
            return result_of(raw)

        def moved(src, dst):
            session = getattr(pipeline, "_session", None)
            _repoint([getattr(session, "current_output_paths", None),
                      getattr(pipeline, "_chain_output_paths", None)], str(src), str(dst))

        def go():
            with _RUN_LOCK:
                rg.set_runner(runner)
                rg.set_move_listener(moved)
                try:
                    return core(paths, changes, new_seed, False, confirm_cost)
                finally:
                    rg.set_runner(None)
                    rg.set_move_listener(None)
        return await asyncio.to_thread(go)

    regenerate_from_source.__doc__ = rg.regenerate_from_source.__doc__
    return tool_decorator(regenerate_from_source)
