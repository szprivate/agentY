"""
agentY – Post-Brain workflow executor.

After the Brain assembles and validates a ComfyUI workflow it calls
``signal_workflow_ready(workflow_path)``.  The pipeline then calls
``execute_workflow()`` (single) or ``execute_workflows_batch()`` (batch), which:

1. Submits the workflow(s) to ComfyUI (``POST /prompt``).
   Batch: ALL workflows are submitted before any polling begins, so
   ComfyUI can start working on the queue immediately.
2. Polls until execution completes (zero LLM tokens burned during the wait).
   Batch: all jobs are monitored CONCURRENTLY, so a successful member streams
   and collects its outputs without waiting on slower/failing siblings. When a
   member fails and a ``repair_fn`` is supplied, it is healed concurrently
   (bounded) and re-queued on the fly, while the survivors keep running.
3. Copies every output file from ComfyUI's configured output directory to
   the path specified in the query templates' brainbriefing (``output_nodes[].output_path``).
   Falls back to downloading via ``/view`` when the output directory cannot be
   determined from the ComfyUI API.
4. When the user has a QA briefing in force, judges every produced file against
   it with the qa_checker agent — and, in a batch, re-generates a failing member
   against the criteria it missed (bounded by ``qa.max_retries``), the same
   on-the-fly way a broken member is healed. See ``src/utils/qa.py``.

Usage
-----
    async for status_line in execute_workflow(path, brainbriefing_json):
        print(status_line)

    async for status_line in execute_workflows_batch(paths, brainbriefing_json):
        print(status_line)

Both functions are ``AsyncGenerator[str, None]`` so the pipeline can forward
each status update to the UI in real time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

logger = logging.getLogger("agentY.executor")


# ---------------------------------------------------------------------------
# Execution-error mailbox
#
# When a ComfyUI job fails, the WebSocket streamer produces a *structured*
# failure (node id/type, exception, traceback). The executor still yields a
# human-readable error line to the UI, but it ALSO records the structured
# failure here so the orchestrator can read it after the batch and drive a
# bounded diagnose-and-fix retry — without every executor consumer having to
# know about a new yield type. Mirrors the workflow-signal mailbox pattern:
# single event loop, one turn at a time, so a module-level list is safe.
# ---------------------------------------------------------------------------
_exec_errors: list[dict] = []


def _record_exec_error(details: dict | None, workflow_path: str = "", error: str = "") -> None:
    """Append one structured ComfyUI execution failure to the mailbox."""
    _exec_errors.append({
        "details": details or {},
        "workflow_path": workflow_path,
        "error": error or "ComfyUI execution failed",
    })


def get_and_clear_exec_errors() -> list[dict]:
    """Return the recorded execution errors and clear the mailbox."""
    out = list(_exec_errors)
    _exec_errors.clear()
    return out


def clear_exec_errors() -> None:
    """Drop any recorded execution errors (call before a fresh run)."""
    _exec_errors.clear()


def _project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def _load_config() -> dict:
    from src.utils.settings import load_settings
    return load_settings()


def _autoload_workflows_into_canvas() -> bool:
    """Whether to mirror each executed workflow onto the ComfyUI canvas.

    Priority: env ``AGENTY_CANVAS_AUTOLOAD`` (1/0) overrides; otherwise the
    ``autoload_workflows_into_canvas`` setting in settings.json (default False).
    Off by default — workflows still build and run, but the graph is only loaded
    onto the canvas when the user asks (the agent offers to).
    """
    env = os.environ.get("AGENTY_CANVAS_AUTOLOAD")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(_load_config().get("autoload_workflows_into_canvas", False))


def _console_lines() -> bool | None:
    """Whether to relay ComfyUI's own terminal output into the run stream.

    Priority, as everywhere else: env ``AGENTY_COMFY_CONSOLE`` (1/0) wins,
    otherwise ``comfyui_console_lines`` in settings (default on). None hands
    the decision back to agenty_core, which is what resolves the env var — so
    "env is set" is expressed by declining to answer.
    """
    if os.environ.get("AGENTY_COMFY_CONSOLE", "").strip():
        return None
    val = _load_config().get("comfyui_console_lines")
    return None if val is None else bool(val)


def _output_dir() -> Path:
    """Return the fallback directory where ComfyUI output files are saved."""
    cfg = _load_config()
    od = cfg.get("output_dir", "./output/")
    return (_project_root() / od).resolve()


# --- ComfyUI dir cache -----------------------------------------------------
# /system_stats returns ComfyUI's argv, which is constant for the lifetime of
# the server process.  Both --output-directory and --user-directory are parsed
# from a single response and memoised so per-output-file resolution doesn't
# trigger a new HTTP roundtrip every call.  Reset via _reset_comfyui_dir_cache
# (e.g. when the user restarts ComfyUI from the agent UI).
_COMFYUI_DIR_CACHE_LOADED: bool = False
_COMFYUI_OUTPUT_DIR: Path | None = None
_COMFYUI_USER_DIR: Path | None = None


def _reset_comfyui_dir_cache() -> None:
    global _COMFYUI_DIR_CACHE_LOADED, _COMFYUI_OUTPUT_DIR, _COMFYUI_USER_DIR
    _COMFYUI_DIR_CACHE_LOADED = False
    _COMFYUI_OUTPUT_DIR = None
    _COMFYUI_USER_DIR = None


def _load_comfyui_dirs() -> None:
    global _COMFYUI_DIR_CACHE_LOADED, _COMFYUI_OUTPUT_DIR, _COMFYUI_USER_DIR
    if _COMFYUI_DIR_CACHE_LOADED:
        return
    try:
        from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag

        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        out_dir = parse_argv_dir_flag(argv, "--output-directory")
        if out_dir:
            _COMFYUI_OUTPUT_DIR = Path(out_dir).resolve()
        usr_dir = parse_argv_dir_flag(argv, "--user-directory")
        if usr_dir:
            _COMFYUI_USER_DIR = Path(usr_dir).resolve()
    except Exception as exc:
        logger.debug("executor: could not query ComfyUI dirs — %s", exc)
    _COMFYUI_DIR_CACHE_LOADED = True


def _get_comfyui_output_dir() -> Path | None:
    """Return ComfyUI's --output-directory (cached for the process lifetime)."""
    _load_comfyui_dirs()
    return _COMFYUI_OUTPUT_DIR


def _get_comfyui_user_dir() -> Path | None:
    """Return ComfyUI's --user-directory (cached for the process lifetime)."""
    _load_comfyui_dirs()
    return _COMFYUI_USER_DIR


# _archive_input_images removed: input files are uploaded to ComfyUI via the
# upload_image tool and live in ComfyUI's --input-directory; no secondary copy
# is needed.  The upload filename is captured in the conversation history and
# summarised into INPUT_PATHS so subsequent sessions can reference it.


def _copy_workflow_to_user_dir(workflow_path: str) -> None:
    """Ensure the finished workflow JSON is in ComfyUI's workflow browser.

    Destination: ``{user_dir}/default/workflows/agentY/`` — the ``default``
    profile segment matters, because that is the only place ComfyUI's workflow
    browser reads. (This used to copy to ``{user_dir}/workflows/``, a sibling
    folder the browser never lists, so agent workflows were effectively
    invisible.)

    Normally a no-op now: ``_workflows_dir()`` writes there in the first place.
    It still earns its keep when that fell back to the in-repo directory because
    ComfyUI was unreachable at assembly time. Silently skips when the source is
    missing or already at the destination.
    """
    import shutil

    src = Path(workflow_path)
    if not src.exists():
        logger.debug("executor: _copy_workflow_to_user_dir: source not found: %s", workflow_path)
        return

    user_dir = _get_comfyui_user_dir()
    if user_dir is None:
        cfg = _load_config()
        fallback = cfg.get("comfyui_user_dir", "")
        if fallback:
            user_dir = Path(fallback).resolve()
        else:
            logger.debug("executor: _copy_workflow_to_user_dir: no user dir configured, skipping")
            return

    dest_dir = user_dir / "default" / "workflows" / "agentY"
    try:
        if src.resolve().parent == dest_dir.resolve():
            return  # already written straight to the browser's folder
    except Exception:  # noqa: BLE001 — resolve can fail on odd paths; just copy
        pass
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        shutil.copy2(src, dest)
        logger.info("executor: workflow copied to user dir → %s", dest)
    except Exception as exc:
        logger.warning("executor: could not copy workflow to user dir — %s", exc)


# _resolve_brainbriefing_output_dir removed: output files are now always kept
# in ComfyUI's --output-directory; _resolve_output_path returns their
# authoritative on-disk path directly from /system_stats without copying.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_node_titles(workflow_path: str) -> dict[str, str]:
    """Return a mapping of node_id -> display name for the given workflow.

    The display name is ``_meta.title`` when present, otherwise ``class_type``.
    Returns an empty dict on any error.
    """
    try:
        p = Path(workflow_path)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        titles: dict[str, str] = {}
        for node_id, node_data in data.items():
            if not isinstance(node_data, dict):
                continue
            title = node_data.get("_meta", {}).get("title", "") or node_data.get("class_type", "")
            if title:
                titles[str(node_id)] = title
        return titles
    except Exception:
        return {}


def _fit_to_briefing(workflow_path: str, qa_briefing) -> tuple:
    """Make the graph satisfy the briefing's measurable requirements up front.

    QA can only ever report that an output came out the wrong shape — after the
    generation is paid for, and to a retry whose levers (seed, prompt) cannot
    change a shape at all. But the requirement is knowable *now*: the briefing
    says 16:9, the graph says 1024x1024, and one parameter decides which wins.

    So the check that would have failed is answered before submission. Returns
    ``(path_to_submit, lines)`` — a SIBLING file when anything changed, so the
    workflow the user chose is left exactly as it was for comparison.

    Never raises and never blocks: a graph it cannot read, or a requirement it
    cannot place, submits unchanged and is judged afterwards as before.
    """
    lines: list[str] = []
    technical = getattr(qa_briefing, "technical", None)
    if not technical:
        # Same answer either way; the point is not reading and parsing the
        # workflow on every run that has no measurable requirement to meet.
        return workflow_path, lines
    try:
        import random as _random

        from src.utils.qa_repair import apply_fix, describe_fix, plan_fixes

        src_path = Path(workflow_path)
        graph = json.loads(src_path.read_text(encoding="utf-8"))
        fixes, problems = plan_fixes(graph, technical)
        for problem in problems:
            control = problem["control"]
            lines.append(f"⚠️ Your briefing asks for {control.replace('_', ' ')} "
                         f"{technical.get(control)!r}: {problem['why']} — "
                         "it will be judged on what comes out.")
        if not fixes:
            return workflow_path, lines
        for fix in fixes:
            if apply_fix(graph, fix):
                lines.append(f"📐 Fitted to your briefing — {describe_fix(fix)}")
        out_path = src_path.with_name(f"{src_path.stem}.fit{_random.randint(1000, 9999)}.json")
        out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        return str(out_path), lines
    except Exception as exc:  # noqa: BLE001 — a fit is a courtesy, never a gate
        logger.debug("executor: could not fit the graph to the briefing — %s", exc)
        return workflow_path, lines


def _submit_workflow(workflow_path: str, client_id: str = "") -> str:
    """Submit *workflow_path* to ComfyUI and return the ``prompt_id``.

    When *client_id* is provided it is forwarded so the matching WebSocket
    connection receives this prompt's progress events.

    Raises ``RuntimeError`` on failure.
    """
    from agenty_core.utils.comfyui_client import get_client

    p = Path(workflow_path)
    if not p.exists():
        raise RuntimeError(f"Workflow file not found: {workflow_path}")

    workflow = json.loads(p.read_text(encoding="utf-8"))
    client = get_client()
    payload: dict = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}

    # Mirror the exact workflow onto the ComfyUI canvas so the user sees what
    # actually ran. Off by default (autoload_workflows_into_canvas in
    # settings.json / AGENTY_CANVAS_AUTOLOAD env); best-effort and non-fatal. The
    # open_workflow_in_canvas tool stays available for on-demand ("show me the
    # workflow") calls by the agent when the user asks.
    if _autoload_workflows_into_canvas():
        try:
            from agenty_core.tools.comfyui import open_workflow_in_canvas as _canvas
            _canvas(workflow_path, name=p.stem)
        except Exception:  # noqa: BLE001
            pass

    result = client.post("/prompt", json_data=payload)
    if isinstance(result, dict) and "prompt_id" in result:
        return result["prompt_id"]
    raise RuntimeError(f"Unexpected response from ComfyUI /prompt: {result!r}")


def _free_vram_for_comfyui() -> None:
    """Best-effort: ask Ollama to evict any resident models before ComfyUI runs.

    Called once per executor invocation (single or batch) — *not* per workflow —
    so a 5-iteration batch doesn't trigger 5 unload roundtrips.  ``/api/ps``
    returns immediately when nothing is loaded, which is the common case for
    pure-Anthropic sessions.
    """
    try:
        from src.tools.agent_control import unload_ollama_models
        unload_ollama_models()
    except Exception as exc:
        logger.debug("executor: Ollama unload attempt skipped/failed: %s", exc)


def _clear_comfyui_history() -> None:
    """Wipe ComfyUI's execution history before submitting this run's workflows.

    ComfyUI's ``/history`` accumulates every past prompt's outputs; without this,
    the agent can't tell whether an image in the history belongs to the current
    generation or a previous one. Clearing it right before submission scopes the
    history to this run. Wipes only the completed-history records — NOT the queue
    (clearing the queue would cancel other jobs). Best-effort; non-fatal.
    """
    try:
        from agenty_core.utils.comfyui_client import get_client
        get_client().post("/history", json_data={"clear": True})
        logger.debug("executor: cleared ComfyUI history before submission")
    except Exception as exc:  # noqa: BLE001
        logger.debug("executor: could not clear ComfyUI history: %s", exc)


def _extract_output_files(history: dict) -> list[dict]:
    """Return a flat list of ``{"filename", "subfolder", "type", "node_id"}`` dicts
    from a stripped history response.

    Handles the ``_strip_history`` output format where outputs are nested under
    ``{prompt_id: {"outputs": {node_id: {"images": [...], "gifs": [...], ...}}}}`.
    """
    files: list[dict] = []
    for _prompt_id, entry in history.items():
        if not isinstance(entry, dict):
            continue
        for node_id, node_out in entry.get("outputs", {}).items():
            if not isinstance(node_out, dict):
                continue
            # ComfyUI may use different keys depending on the output node type
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []):
                    if isinstance(item, dict) and "filename" in item:
                        files.append({**item, "node_id": str(node_id)})
    return files


def _resolve_output_path(
    filename: str,
    subfolder: str = "",
    image_type: str = "output",
    fallback_dir: "Path | None" = None,
) -> Path:
    """Return the authoritative on-disk path for a ComfyUI output file.

    Files are **never copied**.  Resolution order:

    1. ComfyUI's configured ``--output-directory`` (queried via ``/system_stats``).
       If the file exists there it is returned as-is so that ``collected_paths``
       always reflects the real server location.
    2. Falls back to downloading via ``/view`` into *fallback_dir* when supplied
       (taken from ``output_nodes[].output_path`` in the brainbriefing), or into
       the agent's ``output_dir`` (from settings.json) as a last resort.

    This means ``OUTPUT_PATHS`` in the compressed summary are always real,
    accessible paths that the next session's Researcher can pass directly to
    ``upload_image()``.
    """
    # --- try the ComfyUI output dir on disk ------------------------------------
    comfy_out = _get_comfyui_output_dir()
    if comfy_out is not None:
        src = comfy_out / subfolder / filename if subfolder else comfy_out / filename
        if src.exists():
            logger.info("executor: output located at %s (%d bytes)", src, src.stat().st_size)
            return src
        logger.debug(
            "executor: %s not found in ComfyUI output dir, falling back to /view", src
        )

    # --- fallback: download via /view to local output_dir ---------------------
    from agenty_core.utils.comfyui_client import get_client

    if fallback_dir is None:
        fallback_dir = _output_dir()
    fallback_dir.mkdir(parents=True, exist_ok=True)
    dest = fallback_dir / filename

    params: dict = {"filename": filename, "type": image_type}
    if subfolder:
        params["subfolder"] = subfolder

    client = get_client()
    resp = client.get("/view", params=params, raw=True)
    image_bytes: bytes = resp.content  # type: ignore[attr-defined]
    dest.write_bytes(image_bytes)
    logger.info("executor: downloaded output → %s (%d bytes)", dest, len(image_bytes))
    return dest


# ---------------------------------------------------------------------------
# Shared post-processing helper
# ---------------------------------------------------------------------------

async def _process_completed_job(
    history: dict,
    prompt_id: str,
    brainbriefing: dict,
    *,
    workflow_path: str = "",
    user_message: str = "",
    verbose: bool,
    collected_paths: list[str] | None,
    label: str = "",
    qa_briefing=None,
) -> AsyncGenerator[str, None]:
    """Download outputs, run output QA, and collect outputs for one finished job.

    Yields one-line status strings.  ``label`` is an optional prefix like
    ``"[2/5] "`` used in batch runs so the user knows which iteration each
    message belongs to.

    ``user_message`` is the raw text the user originally sent; it is handed to the
    QA agent as the request being judged and never enters any other agent's
    context window.

    *qa_briefing* is a :class:`src.utils.qa.QaBriefing` — the user's own criteria
    and reference images. QA runs only when one is present: with no briefing there
    is no standard to judge against, and inventing one is how a checker ends up
    failing work for not matching its own taste. On a failing verdict this yields a
    single ``{"qa_fail": …}`` dict for the caller to act on.
    """
    pfx = label  # e.g. "[2/5] " or ""

    # NOTE: the workflow's own input images are deliberately NOT fed to QA as
    # references. The old checker did that to score "edit fidelity", which fails
    # every legitimate transformation — "turn this photo into a watercolour" is
    # *supposed* to differ from its input. The briefing decides what the output is
    # compared against; wire the input into the QA hook if that's what you want.
    output_files = _extract_output_files(history)
    if not output_files:
        yield f"{pfx}⚠️ No output files found in ComfyUI history."
        logger.warning("executor: no output files in history for prompt_id=%s", prompt_id)
        return

    # Build a node_id → fallback_dir map from the brainbriefing output_nodes so
    # that downloaded outputs land in the task-specific directory the Researcher
    # chose, rather than the generic agent output_dir.
    _bb_output_dirs: dict[str, Path] = {}
    try:
        for on in brainbriefing.get("output_nodes", []):
            if not isinstance(on, dict):
                continue
            nid = str(on.get("node_id", ""))
            op = on.get("output_path", "")
            if nid and op:
                p = Path(op)
                if not p.is_absolute():
                    p = _output_dir() / p
                _bb_output_dirs[nid] = p
    except Exception as exc:
        logger.debug("executor: could not parse output_nodes from brainbriefing — %s", exc)

    # Resolve each output file's authoritative on-disk path (no copying).
    # Each path is appended to ``collected_paths`` immediately so the caller
    # (chainlit) can flush the image to the UI as soon as the "💾 Output:" line
    # is yielded, instead of waiting for the whole batch to finish.
    #
    # Primary resolution (no network call): because apply_brainbriefing sets
    # filename_prefix = output_path (e.g. "W:/.../output/image_generation"),
    # ComfyUI saves files as output_path + "_00001_.png", i.e. at
    # Path(output_path).parent / filename.  We check that location first and
    # only fall back to _resolve_output_path (which hits /system_stats) when
    # the file is not found there.
    saved_paths: list[Path] = []
    for item in output_files:
        filename = item.get("filename", "")
        subfolder = item.get("subfolder", "")
        file_type = item.get("type", "output")
        node_id = item.get("node_id", "")
        if not filename:
            continue
        fallback_dir = _bb_output_dirs.get(node_id)
        try:
            resolved: Path | None = None
            # Primary: derive path directly from the brainbriefing output_path.
            if fallback_dir is not None:
                bb_candidate = fallback_dir.parent / filename
                if bb_candidate.exists():
                    logger.info(
                        "executor: output at brainbriefing path %s (%d bytes)",
                        bb_candidate, bb_candidate.stat().st_size,
                    )
                    resolved = bb_candidate
            # Fallback: query ComfyUI /system_stats or download via /view.
            if resolved is None:
                resolved = _resolve_output_path(filename, subfolder, file_type, fallback_dir=fallback_dir)
            saved_paths.append(resolved)
            if collected_paths is not None:
                collected_paths.append(str(resolved))
            # Which member produced this file. Batch members are monitored
            # concurrently and a healed one is re-queued, so the order files
            # arrive in is not the order they were submitted in — this is the
            # only place the two are known together. See src.utils.output_tags.
            if workflow_path:
                try:
                    from src.utils.output_tags import note_source
                    note_source(str(resolved), workflow_path)
                except Exception:  # noqa: BLE001
                    pass
            yield f"{pfx}💾 Output: `{resolved}`"
        except Exception as exc:
            yield f"{pfx}⚠️ Could not resolve `{filename}`: {exc}"
            logger.warning("executor: resolve failed for %s — %s", filename, exc)

    if not saved_paths:
        yield f"{pfx}❌ All output downloads failed."
        return

    # Output QA — only against the user's own briefing (see src/utils/qa.py).
    qa_failures: list[dict] = []
    if qa_briefing:
        from src.utils.qa import check_output, qa_settings

        cfg = qa_settings()
        # Files an `agentY qa` node named through its `judge` input — a collector,
        # a loader, a path — are judged alongside what this run produced, not
        # instead of it. `judge` says which outputs a briefing is ABOUT; reading
        # it as "only these" would let one mis-wire quietly excuse everything else
        # from being checked, and an unchecked output is the failure QA exists to
        # prevent. Already-seen paths are dropped, so a collector holding this
        # run's own output does not get judged twice and reported as two failures.
        judgeable = qa_briefing.outputs_with(saved_paths)
        checked = judgeable[:cfg["max_outputs"]]
        skipped = len(judgeable) - len(checked)
        named = len(judgeable) - len(saved_paths)
        yield (f"{pfx}🔍 QA — {qa_briefing.describe()}"
               + (f" · +{named} named by `judge`" if named > 0 else "")
               + (f" · checking {len(checked)} of {len(judgeable)} outputs" if skipped else ""))
        # One agent for the whole job: constructing it costs a model handshake, and
        # it is wiped between outputs anyway so each is still judged on its own.
        agent = None
        try:
            from src.agent import create_qa_agent
            agent = create_qa_agent()
        except Exception as exc:  # noqa: BLE001
            yield f"{pfx}⚠️ QA agent unavailable ({exc}) — delivering outputs unchecked."
        if agent is not None:
            for path in checked:
                result = await asyncio.to_thread(
                    check_output, str(path), qa_briefing,
                    request=user_message, agent=agent,
                )
                yield f"{pfx}🔍 {result.render()}"
                if result.error:
                    continue  # unreadable judge — never counts against the output
                if not result.passed:
                    qa_failures.append({
                        "path": str(path),
                        "summary": result.summary,
                        "failed": result.failed_criteria(),
                    })

    # Copy the finished workflow to the ComfyUI user directory — run in a
    # background thread so a slow UNC/network share doesn't stall the pipeline.
    if workflow_path:
        import threading as _threading
        _threading.Thread(
            target=_copy_workflow_to_user_dir,
            args=(workflow_path,),
            daemon=True,
            name="copy-workflow-to-user-dir",
        ).start()

    if qa_failures:
        # Hand the failure up. The batch layer re-generates against the failed
        # criteria (bounded by qa.max_retries); the single-workflow path surfaces
        # it. The outputs are NOT withheld either way — a failed verdict is an
        # opinion about the user's own criteria, not a reason to hide their file.
        yield {
            "qa_fail": True,
            "workflow_path": workflow_path,
            "image_paths": [str(p) for p in saved_paths],
            "fail_details": qa_failures,
        }
        return

    output_summary = ", ".join(f"`{p.name}`" for p in saved_paths)
    yield f"{pfx}✅ Done. Outputs: {output_summary}"
    if verbose:
        print(f"[executor] {pfx}Finished. Outputs: {[str(p) for p in saved_paths]}")


# ---------------------------------------------------------------------------
# Public executor — single workflow
# ---------------------------------------------------------------------------

async def execute_workflow(
    workflow_path: str,
    brainbriefing_json: str,
    *,
    user_message: str = "",
    verbose: bool = True,
    collected_paths: list[str] | None = None,
    qa_briefing=None,
) -> AsyncGenerator[str, None]:
    """Submit the validated workflow, poll ComfyUI, run QA, and collect outputs.

    This is an ``AsyncGenerator[str, None]`` — each yielded string is a one-line
    status update that the pipeline can forward to the UI as a streaming event.

    Unlike the batch path this does NOT re-generate on a QA failure: it is used for
    chained stages (``run_workflow_now``, ``iterate_step``) where the caller is an
    agent that is still in the loop and can decide for itself what to do about a
    verdict. The failing dict is yielded through for it to see.

    Args:
        workflow_path:      Absolute path to the validated workflow JSON.
        brainbriefing_json: The Query Templates' brainbriefing as a JSON string.
        user_message:       The raw text the user originally sent. Handed to the
                            QA agent as the request being judged; never added to
                            any other agent's conversation history.
        verbose:            Log progress to stdout when True.
        qa_briefing:        Optional :class:`src.utils.qa.QaBriefing` to judge the
                            produced files against. None (the default) = no QA.
    """
    import uuid

    from agenty_core.utils.comfyui_progress import stream_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    # ── 1. Submit ──────────────────────────────────────────────────────────
    _free_vram_for_comfyui()
    _clear_comfyui_history()  # scope history to this run (no stale prior outputs)
    workflow_path, _fit_lines = _fit_to_briefing(workflow_path, qa_briefing)
    for _line in _fit_lines:
        yield _line
    yield "🚀 Submitting workflow to ComfyUI…"
    client_id = uuid.uuid4().hex
    try:
        prompt_id = _submit_workflow(workflow_path, client_id=client_id)
    except Exception as exc:
        error_msg = f"❌ ComfyUI submission failed: {exc}"
        logger.error("executor: %s", error_msg)
        yield error_msg
        return

    yield f"✅ Queued · prompt_id=`{prompt_id}` — streaming progress…"
    if verbose:
        print(f"[executor] Queued prompt_id={prompt_id}")

    # ── 2. Stream progress via WebSocket ───────────────────────────────────
    node_titles = _load_node_titles(workflow_path)
    history: dict | None = None
    error_result: dict | None = None
    _gen = stream_comfyui_job(prompt_id, client_id, node_titles=node_titles,
                              console=_console_lines())
    try:
        async for event in _gen:
            if isinstance(event, dict):
                if "history" in event:
                    history = event["history"]
                else:
                    error_result = event
                break
            yield event
    finally:
        await _gen.aclose()

    if error_result is not None:
        if error_result.get("interrupted"):
            # A deliberate stop, not a defect — do NOT record it or repair it.
            yield "🛑 Execution interrupted — not a workflow error; nothing to repair."
            return
        error_msg = f"❌ ComfyUI execution error: {error_result.get('error')}"
        logger.error("executor: %s", error_msg)
        yield error_msg
        # Record the *structured* failure (node id/type, exception, traceback) to
        # the mailbox so the orchestrator can auto-fix; the line above still shows.
        _record_exec_error(error_result.get("details"), workflow_path,
                           error_result.get("error", ""))
        return

    if history is None:
        yield "❌ ComfyUI stream ended without a result."
        return

    yield "✅ ComfyUI execution complete — collecting outputs…"

    # ── 3-5. Download, QA ─────────────────────────────────────────────────
    async for line in _process_completed_job(
        history,
        prompt_id,
        brainbriefing,
        workflow_path=workflow_path,
        user_message=user_message,
        verbose=verbose,
        collected_paths=collected_paths,
        qa_briefing=qa_briefing,
    ):
        yield line


# ---------------------------------------------------------------------------
# Public executor — batch (submit-all → monitor concurrently → heal failures)
# ---------------------------------------------------------------------------

async def execute_workflows_batch(
    workflow_paths: list[str],
    brainbriefing_json: str,
    *,
    user_message: str = "",
    verbose: bool = True,
    collected_paths: list[str] | None = None,
    qa_briefing=None,
    qa_retry_fn: "Callable[[str, dict], Awaitable[dict]] | None" = None,
    repair_fn: "Callable[[str, dict], Awaitable[dict]] | None" = None,
    max_heal_attempts: int = 3,
    max_concurrent_repairs: int = 3,
    qa_verdicts: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Submit ALL workflows, monitor them CONCURRENTLY, and heal failures on the fly.

    Optimistic parallel execution: every workflow is queued to ComfyUI up front,
    then all jobs are monitored at once so a successful member streams and
    collects its outputs without waiting on slower or failing siblings.  The
    instant a member fails, if ``repair_fn`` is supplied it is repaired
    concurrently (bounded by ``max_concurrent_repairs``) *while the survivors keep
    running*, and the healed workflow is re-queued immediately — up to
    ``max_heal_attempts`` heals per member.  Members that still can't be healed are
    recorded as execution errors for the caller to surface; successful outputs are
    never discarded.

    With ``repair_fn=None`` the behaviour is plain concurrent monitoring: every
    failure is recorded (no healing).

    Args:
        workflow_paths:     Ordered list of absolute workflow JSON file paths.
        brainbriefing_json: Query Templates brainbriefing (for Vision QA).
        user_message:       The raw text the user originally sent.  Forwarded
                            to the Vision QA agent as the ground-truth reference.
                            Never added to any agent's conversation history.
        verbose:            Log progress to stdout when True.
        qa_briefing:        Optional :class:`src.utils.qa.QaBriefing`. When set,
                            every produced file is judged against it.
        qa_retry_fn:        Optional ``async (workflow_path, qa_fail) -> dict``
                            returning ``{"status": "ready"|..., "workflow_path": ...}``.
                            Called with the criteria an output MISSED so the caller
                            can adjust the workflow in place; the member is then
                            re-queued, bounded by ``qa.max_retries``. A QA failure
                            is a different animal from an execution failure — the
                            graph ran fine, the picture is just wrong — so it gets
                            its own budget rather than eating the heal budget.
        repair_fn:          Optional ``async (workflow_path, exec_error) -> dict``
                            returning ``{"status": "ready"|..., "workflow_path": ...}``.
                            Called to heal a failed member's file in place.
        max_heal_attempts:  Per-member heal cap (default 3).
        max_concurrent_repairs: Max repairs running at once (default 3).
    """
    import uuid

    from agenty_core.utils.comfyui_progress import stream_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    total = len(workflow_paths)
    labels = {wf: f"[{i}/{total}] " for i, wf in enumerate(workflow_paths, 1)}

    def _label(wf: str) -> str:
        return labels.get(wf, "")

    # Concurrent monitor/heal tasks funnel their output lines and terminal
    # control events through this queue; the generator body below drains it.
    out_q: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
    repair_sem = asyncio.Semaphore(max(1, max_concurrent_repairs))
    active: set[asyncio.Task] = set()
    # QA re-generations, counted per workflow file and capped by the qa settings.
    # A retry writes an adjusted workflow (possibly under a new path), so the count
    # travels with the file the retry produced.
    qa_retries: dict[str, int] = {}
    try:
        from src.utils.qa import qa_settings
        qa_max_retries = qa_settings()["max_retries"] if qa_briefing else 0
        # A briefing may set its own budget ("retry: 2"). The user writing the
        # criteria is better placed than a global setting to say how many times
        # this particular thing is worth paying for again.
        own = getattr(qa_briefing, "retry_budget", None) if qa_briefing else None
        if own is not None:
            qa_max_retries = max(0, int(own))
    except Exception:  # noqa: BLE001
        qa_max_retries = 0

    def _spawn(coro) -> None:
        t = asyncio.create_task(coro)
        active.add(t)
        t.add_done_callback(active.discard)

    async def _monitor_member(wf_path: str, prompt_id: str, cid: str, heals: int,
                              submit_error: dict | None = None) -> None:
        """Stream one queued job to completion; emit lines + a terminal event."""
        label = _label(wf_path)
        try:
            if submit_error is not None or not prompt_id:
                await out_q.put(("member", ("fail", wf_path, heals,
                                            submit_error or {"details": {}, "error": "submission failed"})))
                return
            history: dict | None = None
            error_result: dict | None = None
            node_titles = _load_node_titles(wf_path)
            gen = stream_comfyui_job(prompt_id, cid, node_titles=node_titles,
                                     console=_console_lines())
            try:
                async for event in gen:
                    if isinstance(event, dict):
                        if "history" in event:
                            history = event["history"]
                        else:
                            error_result = event
                        break
                    # ComfyUI has one console for the whole process, and exactly
                    # one member relays it — so tagging those lines "[2/5]"
                    # would attribute a shared log to whichever member happens
                    # to hold the relay. They also carry their own marker, which
                    # a member prefix would hide from the panel's classifier.
                    await out_q.put(("line", event if event.startswith("🖥")
                                             else f"{label}{event}"))
            finally:
                await gen.aclose()

            if error_result is not None:
                if error_result.get("interrupted"):
                    # Deliberate stop, not a defect — retire the member without
                    # recording an error or triggering a heal.
                    await out_q.put(("line", f"{label}🛑 Execution interrupted — "
                                             f"skipping (not a workflow error)."))
                    await out_q.put(("member", ("interrupted", wf_path, heals, None)))
                    return
                # The monitor already printed the node's own message when it had
                # one ("❌ Error in <node>: <why>"); `error` is a placeholder next
                # to it ("ComfyUI execution failed"), and printing both buries the
                # useful line under a generic one. Say something only when there
                # is something the user hasn't already been told.
                _why = str((error_result.get("details") or {}).get("exception_message") or "")
                if not _why:
                    await out_q.put(("line", f"{label}❌ ComfyUI execution error: "
                                             f"{error_result.get('error')}"))
                logger.error("executor: batch member failed (%s): %s", wf_path,
                             _why or error_result.get("error"))
                await out_q.put(("member", ("fail", wf_path, heals, error_result)))
                return
            if history is None:
                await out_q.put(("line", f"{label}❌ ComfyUI stream ended without a result."))
                await out_q.put(("member", ("fail", wf_path, heals,
                                            {"details": {}, "error": "stream ended without a result"})))
                return
            await out_q.put(("line", f"{label}✅ Complete — collecting outputs…"))
            qa_fail: dict | None = None
            async for line in _process_completed_job(
                history, prompt_id, brainbriefing,
                workflow_path=wf_path, user_message=user_message, verbose=verbose,
                collected_paths=collected_paths, label=label, qa_briefing=qa_briefing,
            ):
                if isinstance(line, dict) and line.get("qa_fail"):
                    qa_fail = line          # judged wrong, not broken — see below
                    continue
                await out_q.put(("line", line))
            if qa_fail is not None:
                await out_q.put(("member", ("qa_fail", wf_path, heals, qa_fail)))
                return
            await out_q.put(("member", ("ok", wf_path, heals, None)))
        except Exception as exc:  # noqa: BLE001
            logger.error("executor: monitor error for %s — %s", wf_path, exc)
            await out_q.put(("line", f"{label}❌ monitor error: {exc}"))
            await out_q.put(("member", ("fail", wf_path, heals, {"details": {}, "error": str(exc)})))

    async def _heal_member(wf_path: str, heals: int, error_result: dict) -> None:
        """Repair a failed member (bounded concurrency), then re-queue it."""
        label = _label(wf_path)
        name = os.path.basename(wf_path)
        async with repair_sem:
            # "Recovering", not "Healing": the callback decides what this member
            # needs, and not everything that fails is a broken graph — a provider
            # refusing the content is re-run rather than repaired.
            await out_q.put(("line", f"{label}🔧 Recovering `{name}` "
                                     f"(attempt {heals + 1}/{max_heal_attempts})…"))
            try:
                res = await repair_fn(wf_path, error_result)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                res = {"status": "failed", "error": str(exc)}
        if (res or {}).get("status") == "rejected":
            # A verdict, not a failed attempt: the graph is fine and running it
            # again has already been tried. Retire the member with the callback's
            # own account of it, which says what the caller can actually do.
            await out_q.put(("line", f"{label}🚫 {(res or {}).get('error') or 'rejected'}"))
            await out_q.put(("heal", ("rejected", wf_path, heals + 1, "", "",
                                      {"details": res, "error": (res or {}).get("error", "")})))
        elif (res or {}).get("status") == "ready":
            cid = uuid.uuid4().hex
            try:
                prompt_id = await asyncio.to_thread(_submit_workflow, wf_path, cid)
                await out_q.put(("line", f"{label}♻️ Re-queued healed `{name}` · prompt_id=`{prompt_id}`"))
                await out_q.put(("heal", ("ready", wf_path, heals + 1, prompt_id, cid, None)))
            except Exception as exc:  # noqa: BLE001
                await out_q.put(("line", f"{label}❌ Re-queue failed for `{name}`: {exc}"))
                await out_q.put(("heal", ("failed", wf_path, heals + 1, "", "",
                                          {"details": {}, "error": str(exc)})))
        else:
            await out_q.put(("line", f"{label}❌ Could not heal `{name}` "
                                     f"({(res or {}).get('error') or 'still invalid'})."))
            await out_q.put(("heal", ("failed", wf_path, heals + 1, "", "", error_result)))

    async def _qa_retry_member(wf_path: str, tries: int, qa_fail: dict) -> None:
        """Re-generate a member that FAILED QA, against the criteria it missed.

        Deliberately parallel to ``_heal_member`` rather than folded into it: that
        one fixes a graph that could not run, this one fixes a graph that ran fine
        and produced the wrong picture. They have different budgets, different
        callbacks, and a user who wants one is not necessarily asking for the other.
        """
        label = _label(wf_path)
        name = os.path.basename(wf_path)
        missed = "; ".join(
            m for d in (qa_fail.get("fail_details") or []) for m in (d.get("failed") or [])
        ) or "the QA briefing"
        async with repair_sem:
            await out_q.put(("line", f"{label}🔁 QA failed — re-generating `{name}` "
                                     f"(attempt {tries + 1}/{qa_max_retries}) against: {missed}"))
            try:
                res = await qa_retry_fn(wf_path, qa_fail)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                res = {"status": "failed", "error": str(exc)}
        new_path = str((res or {}).get("workflow_path") or wf_path)
        if (res or {}).get("status") == "ready":
            cid = uuid.uuid4().hex
            try:
                prompt_id = await asyncio.to_thread(_submit_workflow, new_path, cid)
                await out_q.put(("line", f"{label}♻️ Re-queued `{os.path.basename(new_path)}` "
                                         f"· prompt_id=`{prompt_id}`"))
                qa_retries[new_path] = tries + 1
                labels.setdefault(new_path, label)
                await out_q.put(("qa", ("ready", new_path, prompt_id, cid)))
                return
            except Exception as exc:  # noqa: BLE001
                await out_q.put(("line", f"{label}❌ QA re-queue failed for `{name}`: {exc}"))
        else:
            await out_q.put(("line", f"{label}⚠️ Could not adjust `{name}` for the QA "
                                     f"failure ({(res or {}).get('error') or 'no change made'}) "
                                     f"— delivering the output as it is."))
        await out_q.put(("qa", ("done", wf_path, "", "")))

    # ── Phase 1: submit every workflow, spawn a concurrent monitor for each ─
    _free_vram_for_comfyui()
    _clear_comfyui_history()  # once per batch — scope history to this run only
    outstanding = 0
    for idx, wf_path in enumerate(workflow_paths, 1):
        yield f"🚀 Queuing iteration {idx}/{total}…"
        cid = uuid.uuid4().hex
        try:
            wf_path, _fit_lines = _fit_to_briefing(wf_path, qa_briefing)
            for _line in _fit_lines:
                await out_q.put(("line", f"{label}{_line}"))
            prompt_id = _submit_workflow(wf_path, client_id=cid)
            yield f"✅ Iteration {idx}/{total} queued · prompt_id=`{prompt_id}`"
            if verbose:
                print(f"[executor] Batch {idx}/{total} queued prompt_id={prompt_id}")
            _spawn(_monitor_member(wf_path, prompt_id, cid, 0))
        except Exception as exc:  # noqa: BLE001
            yield f"❌ Submission failed for iteration {idx}/{total}: {exc}"
            logger.error("executor: submission failed for %s — %s", wf_path, exc)
            # Route it through the same failure path so it, too, can be healed.
            _spawn(_monitor_member(wf_path, "", cid, 0,
                                   submit_error={"details": {}, "error": str(exc)}))
        outstanding += 1

    if not outstanding:
        return

    yield (f"⏳ All {total} workflow(s) queued — monitoring concurrently"
           f"{' with self-healing' if repair_fn else ''}…")

    # ── Phase 2: drain events; heal failures the moment they occur ─────────
    # `outstanding` counts live work units. A unit stays alive across a
    # fail→heal→re-queue cycle (member→heal→member) and is only retired when it
    # finally succeeds or exhausts its heal budget.
    try:
        while outstanding > 0:
            kind, payload = await out_q.get()
            if kind == "line":
                yield payload
                continue
            if kind == "member":
                status, wf_path, heals, error_result = payload
                if status == "qa_fail":
                    tries = qa_retries.get(wf_path, 0)
                    if qa_retry_fn is not None and tries < qa_max_retries:
                        _spawn(_qa_retry_member(wf_path, tries, error_result or {}))
                    else:
                        # Budget spent (or retries off): the output stands, and the
                        # verdict has already been shown. Not an execution error —
                        # nothing goes in the error mailbox for the fixer to chase.
                        # It IS recorded, though: a verdict the agent never sees is
                        # a quality gate that only talks to the log.
                        if qa_verdicts is not None:
                            qa_verdicts[wf_path] = {
                                "tries": tries,
                                "outputs": [d.get("path") for d
                                            in ((error_result or {}).get("fail_details") or [])],
                                "missed": [m for d in ((error_result or {}).get("fail_details") or [])
                                           for m in (d.get("failed") or [])],
                                "summary": "; ".join(
                                    str(d.get("summary") or "") for d
                                    in ((error_result or {}).get("fail_details") or []) if d.get("summary")),
                            }
                        outstanding -= 1
                    continue
                if status in ("ok", "interrupted"):
                    # "interrupted" retires the unit like a success: no output to
                    # collect, but no error to record and nothing to heal.
                    outstanding -= 1
                elif repair_fn is not None and heals < max_heal_attempts:
                    _spawn(_heal_member(wf_path, heals, error_result or {}))  # unit persists
                else:
                    _record_exec_error((error_result or {}).get("details"),
                                       wf_path, (error_result or {}).get("error", ""))
                    outstanding -= 1
                continue
            if kind == "heal":
                # `heals` here is the number of repairs completed for this member.
                status, wf_path, heals, prompt_id, cid, error_result = payload
                if status == "ready":
                    _spawn(_monitor_member(wf_path, prompt_id, cid, heals))  # re-run; unit persists
                elif status == "rejected":
                    # Not retried: the callback already decided this cannot be fixed
                    # by trying again (a provider's content refusal, say).
                    _record_exec_error((error_result or {}).get("details"),
                                       wf_path, (error_result or {}).get("error", ""))
                    outstanding -= 1
                elif repair_fn is not None and heals < max_heal_attempts:
                    # Repair couldn't produce a runnable graph — give the fixer
                    # another shot on the in-place file, up to the budget.
                    _spawn(_heal_member(wf_path, heals, error_result or {}))  # unit persists
                else:
                    _record_exec_error((error_result or {}).get("details"),
                                       wf_path, (error_result or {}).get("error", ""))
                    outstanding -= 1
                continue
            if kind == "qa":
                status, wf_path, prompt_id, cid = payload
                if status == "ready":
                    _spawn(_monitor_member(wf_path, prompt_id, cid, 0))  # unit persists
                else:
                    outstanding -= 1  # couldn't adjust — retire with what we have
                continue
    finally:
        for t in list(active):
            t.cancel()
