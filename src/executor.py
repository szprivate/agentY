"""
agentY – Post-Brain workflow executor.

After the Brain assembles and validates a ComfyUI workflow it calls
``signal_workflow_ready(workflow_path)``.  The pipeline then calls
``execute_workflow()`` (single) or ``execute_workflows_batch()`` (batch), which:

1. Submits the workflow(s) to ComfyUI (``POST /prompt``).
   Batch: ALL workflows are submitted before any polling begins, so
   ComfyUI can start working on the queue immediately.
2. Polls until execution completes (zero LLM tokens burned during the wait).
   Batch: polls each prompt_id in submission order; earlier jobs are
   typically already done by the time we reach them.
3. Copies every output file from ComfyUI's configured output directory to
   the path specified in the researcher's brainbriefing (``output_nodes[].output_path``).
   Falls back to downloading via ``/view`` when the output directory cannot be
   determined from the ComfyUI API.
4. Runs a Vision QA pass with an Ollama multimodal model, comparing the
   output against the original brainbriefing.

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

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger("agentY.executor")


def _project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def _load_config() -> dict:
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
    return {}


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
        from src.utils.comfyui_client import get_client

        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        for arg in argv:
            if not isinstance(arg, str):
                continue
            if arg.startswith("--output-directory="):
                _COMFYUI_OUTPUT_DIR = Path(arg.split("=", 1)[1]).resolve()
            elif arg.startswith("--user-directory="):
                _COMFYUI_USER_DIR = Path(arg.split("=", 1)[1]).resolve()
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
    """Copy the finished workflow JSON to the ComfyUI user directory.

    Destination: ``{user_dir}/workflows/``.
    The user directory is resolved from the ``--user-directory=`` flag via
    ``/system_stats``.  Falls back to ``comfyui_user_dir`` in settings.json.
    Silently skips when the workflow file doesn't exist.
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

    dest_dir = user_dir / "workflows"
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


def _load_qa_prompts() -> dict[str, str]:
    """Parse the qa_checker system prompt file into sections.

    The file is divided by ``## <section_name>`` headings.  Returns a dict
    mapping section name → stripped content.  Falls back to empty strings so
    callers always get a valid (possibly empty) value.

    Expected sections: ``system``, ``question_edit``, ``question_generation``.
    """
    cfg = _load_config()
    filename = cfg.get("system_prompts", {}).get("qa_checker", "system_prompt.qaChecker.md")
    config_dir = _project_root() / "config"
    candidate = config_dir / "system_prompts" / filename
    path = candidate if candidate.exists() else config_dir / filename
    if not path.exists():
        logger.warning("executor: QA prompt file not found: %s", path)
        return {}

    import re

    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    # Split on lines that start with '## '
    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    # parts = ['', 'section_name', 'body', 'section_name', 'body', ...]
    it = iter(parts[1:])  # skip leading empty string
    for name, body in zip(it, it):
        sections[name.strip()] = body.strip()
    return sections


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _submit_workflow(workflow_path: str, client_id: str = "") -> str:
    """Submit *workflow_path* to ComfyUI and return the ``prompt_id``.

    When *client_id* is provided it is forwarded so the matching WebSocket
    connection receives this prompt's progress events.

    Raises ``RuntimeError`` on failure.
    """
    from src.utils.comfyui_client import get_client

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


def _extract_output_files(history: dict) -> list[dict]:
    """Return a flat list of ``{"filename", "subfolder", "type"}`` dicts from a
    stripped history response.

    Handles the ``_strip_history`` output format where outputs are nested under
    ``{prompt_id: {"outputs": {node_id: {"images": [...], "gifs": [...], ...}}}}``.
    """
    files: list[dict] = []
    for _prompt_id, entry in history.items():
        if not isinstance(entry, dict):
            continue
        for _node_id, node_out in entry.get("outputs", {}).items():
            if not isinstance(node_out, dict):
                continue
            # ComfyUI may use different keys depending on the output node type
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []):
                    if isinstance(item, dict) and "filename" in item:
                        files.append(item)
    return files


def _resolve_output_path(
    filename: str,
    subfolder: str = "",
    image_type: str = "output",
) -> Path:
    """Return the authoritative on-disk path for a ComfyUI output file.

    Files are **never copied**.  Resolution order:

    1. ComfyUI's configured ``--output-directory`` (queried via ``/system_stats``).
       If the file exists there it is returned as-is so that ``collected_paths``
       always reflects the real server location.
    2. Falls back to downloading via ``/view`` into the agent's ``output_dir``
       (from settings.json) when ComfyUI is remote or the file is not on disk.

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
    from src.utils.comfyui_client import get_client

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


async def _vision_qa(
    image_path: Path,
    brainbriefing: dict,
    *,
    user_message: str = "",
    input_image_paths: list[Path] | None = None,
) -> str:
    """Run a QA pass comparing *image_path* against the user's original request.

    Uses *user_message* (the raw text the user sent) as the ground-truth
    reference.  When *input_image_paths* is supplied (image-editing task) the
    input images are sent alongside the output so the model can judge edit
    fidelity.  Supports any number of input images.

    Image ordering sent to the model: all input images first (IMAGE 1 … N),
    then the generated output as the last image (IMAGE N+1).

    This is a standalone Ollama call that does NOT touch any agent's context
    window or conversation history.  The model used is defined by
    ``llm.pipeline.executor_vision_model`` in settings.json.

    Returns a short verdict string (PASS / FAIL + explanation).
    Never raises — returns an error description on failure.
    """
    from src.utils.llm_functions import LLMFunctions

    try:
        llm = LLMFunctions.for_vision()
        output_bytes = image_path.read_bytes()

        # Build the reference description – prefer the raw user message; fall
        # back to the brainbriefing task/prompt fields so the QA still works
        # even when user_message was not forwarded.
        reference = user_message.strip()
        if not reference:
            task_desc = brainbriefing.get("task", {}).get("description", "")
            positive_prompt = brainbriefing.get("prompt", {}).get("positive", "")
            reference = f"Task: {task_desc}\nPrompt: {positive_prompt}".strip()

        # Load each input image; skip any that cannot be read.
        input_bytes_list: list[bytes] = []
        if input_image_paths:
            for inp_path in input_image_paths:
                try:
                    input_bytes_list.append(inp_path.read_bytes())
                except Exception as read_exc:
                    logger.warning(
                        "executor: could not read input image %s — %s",
                        inp_path, read_exc,
                    )

        is_edit = bool(input_bytes_list)

        qa_prompts = _load_qa_prompts()
        system = qa_prompts.get("system", "You are a visual QA analyst for AI-generated images.")

        if is_edit:
            n_inputs = len(input_bytes_list)
            output_img_num = n_inputs + 1
            if n_inputs == 1:
                image_description = (
                    f"TWO images: IMAGE 1 is the ORIGINAL input image, "
                    f"IMAGE 2 is the GENERATED output image."
                )
            else:
                input_labels = ", ".join(f"IMAGE {i + 1}" for i in range(n_inputs))
                image_description = (
                    f"{n_inputs + 1} images: {input_labels} are the ORIGINAL input images "
                    f"(in the order provided), IMAGE {output_img_num} is the GENERATED output image."
                )
            template_key = "question_edit"
            question = (
                qa_prompts.get(template_key, "")
                .replace("{{REFERENCE}}", reference)
                .replace("{{IMAGE_DESCRIPTION}}", image_description)
                .replace("{{OUTPUT_IMAGE_NUM}}", str(output_img_num))
            )
        else:
            template_key = "question_generation"
            question = qa_prompts.get(template_key, "").replace("{{REFERENCE}}", reference)

        if not question:
            # Minimal inline fallback if the file is missing
            question = (
                f'The user\'s original request was:\n"{reference}"\n\n'
                "Does this generated image satisfy that request?\n"
                "Reply with: PASS or FAIL, followed by a brief explanation."
            )

        # Send images to the model: inputs first (IMAGE 1…N), output last (IMAGE N+1).
        # vision_chat prepends image_bytes as the first image, so we pass the first
        # input (or the output for generation tasks) as the primary argument and
        # append the remainder.
        if is_edit:
            primary_bytes = input_bytes_list[0]
            extra_images: list[bytes] = input_bytes_list[1:] + [output_bytes]
        else:
            primary_bytes = output_bytes
            extra_images = []

        verdict = await llm.vision_chat(
            question,
            primary_bytes,
            system=system,
            extra_images=extra_images or None,
        )
        logger.info("executor: vision QA for %s → %s", image_path.name, verdict[:120])
        return verdict.strip()

    except Exception as exc:
        logger.warning("executor: vision QA failed for %s — %s", image_path.name, exc)
        return f"Vision QA unavailable: {exc}"


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
    run_qa: bool = False,
) -> AsyncGenerator[str, None]:
    """Download outputs, run Vision QA, and collect outputs for one finished job.

    Yields one-line status strings.  ``label`` is an optional prefix like
    ``"[2/5] "`` used in batch runs so the user knows which iteration each
    message belongs to.

    ``user_message`` is the raw text the user originally sent; it is passed
    straight to ``_vision_qa`` and never enters any agent's context window.
    The input image path (for edit-task fidelity checks) is derived from the
    ``input_nodes`` field of the brainbriefing automatically.
    """
    pfx = label  # e.g. "[2/5] " or ""

    # Resolve all input image paths from the brainbriefing (edit fidelity check).
    input_image_paths: list[Path] = []
    try:
        input_nodes = brainbriefing.get("input_nodes", [])
        if input_nodes and isinstance(input_nodes, list):
            for node in input_nodes:
                raw_path = node.get("path", "") if isinstance(node, dict) else ""
                if raw_path:
                    candidate = Path(raw_path)
                    if candidate.exists():
                        input_image_paths.append(candidate)
                    else:
                        logger.debug(
                            "executor: input_node path does not exist on disk: %s", raw_path
                        )
    except Exception as exc:
        logger.debug("executor: could not resolve input image paths from brainbriefing — %s", exc)

    output_files = _extract_output_files(history)
    if not output_files:
        yield f"{pfx}⚠️ No output files found in ComfyUI history."
        logger.warning("executor: no output files in history for prompt_id=%s", prompt_id)
        return

    # Resolve each output file's authoritative on-disk path (no copying).
    # Each path is appended to ``collected_paths`` immediately so the caller
    # (chainlit) can flush the image to the UI as soon as the "💾 Output:" line
    # is yielded, instead of waiting for the whole batch to finish.
    saved_paths: list[Path] = []
    for item in output_files:
        filename = item.get("filename", "")
        subfolder = item.get("subfolder", "")
        file_type = item.get("type", "output")
        if not filename:
            continue
        try:
            output_path = _resolve_output_path(filename, subfolder, file_type)
            saved_paths.append(output_path)
            if collected_paths is not None:
                collected_paths.append(str(output_path))
            yield f"{pfx}💾 Output: `{output_path}`"
        except Exception as exc:
            yield f"{pfx}⚠️ Could not resolve `{filename}`: {exc}"
            logger.warning("executor: resolve failed for %s — %s", filename, exc)

    if not saved_paths:
        yield f"{pfx}❌ All output downloads failed."
        return

    # Vision QA — only when explicitly requested by the user.
    qa_failures: list[dict] = []
    if run_qa:
        yield f"{pfx}🔍 Running Vision QA…"
        for path in saved_paths:
            verdict = await _vision_qa(
                path,
                brainbriefing,
                user_message=user_message,
                input_image_paths=input_image_paths or None,
            )
            yield f"{pfx}🔍 QA `{path.name}` → {verdict}"
            if "FAIL" in verdict.upper():
                qa_failures.append({"path": str(path), "verdict": verdict})

    # Copy the finished workflow to the ComfyUI user directory for easy reuse.
    if workflow_path:
        _copy_workflow_to_user_dir(workflow_path)

    if qa_failures:
        # Signal the pipeline layer to pause and ask the user.
        yield {
            "qa_fail": True,
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
    run_qa: bool = False,
) -> AsyncGenerator[str, None]:
    """Submit the validated workflow, poll ComfyUI, run QA, and collect outputs.

    This is an ``AsyncGenerator[str, None]`` — each yielded string is a one-line
    status update that the pipeline can forward to the UI as a streaming event.

    Args:
        workflow_path:      Absolute path to the validated workflow JSON.
        brainbriefing_json: The Researcher's brainbriefing as a JSON string,
                            used to extract input image paths for QA comparison.
        user_message:       The raw text the user originally sent.  Forwarded
                            to the Vision QA agent as the ground-truth reference.
                            Never added to any agent's conversation history.
        verbose:            Log progress to stdout when True.
    """
    import uuid

    from src.utils.comfyui_progress import stream_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    # ── 1. Submit ──────────────────────────────────────────────────────────
    _free_vram_for_comfyui()
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
    history: dict | None = None
    error_result: dict | None = None
    async for event in stream_comfyui_job(prompt_id, client_id):
        if isinstance(event, dict):
            if "history" in event:
                history = event["history"]
            else:
                error_result = event
            break
        yield event

    if error_result is not None:
        error_msg = f"❌ ComfyUI execution error: {error_result.get('error')}"
        logger.error("executor: %s", error_msg)
        yield error_msg
        return

    if history is None:
        yield "❌ ComfyUI stream ended without a result."
        return

    yield "✅ ComfyUI execution complete — collecting outputs…"

    # ── 3-5. Download, QA ─────────────────────────────────────────────────
    async for line in _process_completed_job(
        history,
        prompt_id,
        brainbriefing,        workflow_path=workflow_path,        user_message=user_message,
        verbose=verbose,
        collected_paths=collected_paths,
        run_qa=run_qa,
    ):
        yield line


# ---------------------------------------------------------------------------
# Public executor — batch (submit-all-then-poll)
# ---------------------------------------------------------------------------

async def execute_workflows_batch(
    workflow_paths: list[str],
    brainbriefing_json: str,
    *,
    user_message: str = "",
    verbose: bool = True,
    collected_paths: list[str] | None = None,
    run_qa: bool = False,
) -> AsyncGenerator[str, None]:
    """Submit ALL workflows to ComfyUI first, then poll + process each in order.

    This avoids the submit→wait→submit→wait pattern: ComfyUI receives the full
    batch immediately and can start executing jobs while the client is still
    submitting remaining ones.  Polling then starts *after* the last submission,
    so earlier jobs are frequently already done by the time we reach them.

    Args:
        workflow_paths:     Ordered list of absolute workflow JSON file paths.
        brainbriefing_json: Researcher brainbriefing (for Vision QA).
        user_message:       The raw text the user originally sent.  Forwarded
                            to the Vision QA agent as the ground-truth reference.
                            Never added to any agent's conversation history.
        verbose:            Log progress to stdout when True.
    """
    import uuid

    from src.utils.comfyui_progress import stream_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    total = len(workflow_paths)

    # ── Phase 1: submit all ────────────────────────────────────────────────
    _free_vram_for_comfyui()
    # One client_id per prompt so each WebSocket subscription only receives
    # events for its own job.
    queued: list[tuple[str, str, str]] = []  # [(prompt_id, workflow_path, client_id), ...]
    for idx, wf_path in enumerate(workflow_paths, 1):
        yield f"🚀 Queuing iteration {idx}/{total}…"
        cid = uuid.uuid4().hex
        try:
            prompt_id = _submit_workflow(wf_path, client_id=cid)
            queued.append((prompt_id, wf_path, cid))
            yield f"✅ Iteration {idx}/{total} queued · prompt_id=`{prompt_id}`"
            if verbose:
                print(f"[executor] Batch {idx}/{total} queued prompt_id={prompt_id}")
        except Exception as exc:
            error_msg = f"❌ Submission failed for iteration {idx}/{total}: {exc}"
            logger.error("executor: %s", error_msg)
            yield error_msg

    if not queued:
        yield "❌ All workflow submissions failed — nothing to poll."
        return

    yield (
        f"⏳ All {len(queued)}/{total} workflows queued — "
        f"streaming progress in submission order…"
    )

    # ── Phase 2: stream progress + process each in submission order ────────
    for idx, (prompt_id, _wf_path, cid) in enumerate(queued, 1):
        label = f"[{idx}/{len(queued)}] "
        yield f"{label}⏳ Streaming progress (prompt_id=`{prompt_id}`)…"
        if verbose:
            print(f"[executor] Batch streaming {idx}/{len(queued)} prompt_id={prompt_id}")

        history: dict | None = None
        error_result: dict | None = None
        async for event in stream_comfyui_job(prompt_id, cid):
            if isinstance(event, dict):
                if "history" in event:
                    history = event["history"]
                else:
                    error_result = event
                break
            yield f"{label}{event}"

        if error_result is not None:
            error_msg = f"{label}❌ ComfyUI execution error: {error_result.get('error')}"
            logger.error("executor: %s", error_msg)
            yield error_msg
            continue  # move on to the next iteration, don't abort the whole batch

        if history is None:
            yield f"{label}❌ ComfyUI stream ended without a result."
            continue

        yield f"{label}✅ Complete — collecting outputs…"

        async for line in _process_completed_job(
            history,
            prompt_id,
            brainbriefing,
            workflow_path=_wf_path,
            user_message=user_message,
            verbose=verbose,
            collected_paths=collected_paths,
            label=label,
            run_qa=run_qa,
        ):
            yield line
