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
3. Downloads every output image to ``./output/<filename>``.
4. Runs a Vision QA pass with an Ollama multimodal model, comparing the
   output against the original brainbriefing.
5. Sends each image to Slack (using the active channel context).

Usage
-----
    async for status_line in execute_workflow(path, brainbriefing_json,
                                              slack_channel_id=cid,
                                              slack_thread_ts=ts):
        print(status_line)

    async for status_line in execute_workflows_batch(paths, brainbriefing_json,
                                                     slack_channel_id=cid,
                                                     slack_thread_ts=ts):
        print(status_line)

Both functions are ``AsyncGenerator[str, None]`` so the pipeline can forward
each status update to Slack in real time.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger("agentY.executor")

_OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _submit_workflow(workflow_path: str) -> str:
    """Submit *workflow_path* to ComfyUI and return the ``prompt_id``.

    Raises ``RuntimeError`` on failure.
    """
    from src.utils.comfyui_client import get_client

    p = Path(workflow_path)
    if not p.exists():
        raise RuntimeError(f"Workflow file not found: {workflow_path}")

    workflow = json.loads(p.read_text(encoding="utf-8"))
    client = get_client()
    payload: dict = {"prompt": workflow}
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}

    # Best-effort: ask the Ollama server to unload the vision/LLM model
    # before submitting large workflows to ComfyUI so GPUs can be freed.
    try:
        from src.utils.llm_functions import LLMFunctions
        import httpx

        llm_vis = LLMFunctions.for_vision()
        model = llm_vis.model
        host = llm_vis.host.rstrip("/")

        # Try several plausible Ollama unload endpoints; ignore failures.
        unload_paths = [
            f"{host}/api/models/{model}:unload",
            f"{host}/api/models/{model}:stop",
            f"{host}/api/models/{model}/unload",
        ]
        for url in unload_paths:
            try:
                httpx.post(url, timeout=5.0)
                logger.info("executor: requested Ollama unload -> %s", url)
                break
            except Exception:
                continue
    except Exception:
        # Non-fatal: proceed with submission even if unload attempt fails.
        logger.debug("executor: Ollama unload attempt skipped/failed")

    result = client.post("/prompt", json_data=payload)
    if isinstance(result, dict) and "prompt_id" in result:
        return result["prompt_id"]
    raise RuntimeError(f"Unexpected response from ComfyUI /prompt: {result!r}")


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


def _download_output(filename: str, subfolder: str = "", image_type: str = "output") -> Path:
    """Download *filename* from ComfyUI to ``./output/`` and return the local path."""
    from src.utils.comfyui_client import get_client

    params: dict = {"filename": filename, "type": image_type}
    if subfolder:
        params["subfolder"] = subfolder

    client = get_client()
    resp = client.get("/view", params=params, raw=True)
    image_bytes: bytes = resp.content  # type: ignore[attr-defined]

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = _OUTPUT_DIR / filename
    dest.write_bytes(image_bytes)
    logger.info("executor: saved output → %s (%d bytes)", dest, len(image_bytes))
    return dest


async def _vision_qa(
    image_path: Path,
    brainbriefing: dict,
) -> str:
    """Ask an Ollama vision model whether *image_path* satisfies the brief.

    Returns a short QA verdict string.  Never raises — returns an error
    description on failure so the pipeline can continue.
    """
    from src.utils.llm_functions import LLMFunctions

    try:
        llm = LLMFunctions.for_vision()
        image_bytes = image_path.read_bytes()

        task_desc = brainbriefing.get("task", {}).get("description", "")
        positive_prompt = brainbriefing.get("prompt", {}).get("positive", "")
        brief_summary = (
            f"Task: {task_desc}\n"
            f"Prompt: {positive_prompt}"
        ).strip()

        system = (
            "You are a visual QA analyst for AI-generated images. "
            "Be concise — max 3 sentences. Focus on whether the image matches "
            "the brief and note any obvious artifacts or failures."
        )
        question = (
            f"Does this generated image satisfy the following brief?\n\n"
            f"{brief_summary}\n\n"
            "Reply with: PASS or FAIL, followed by a brief explanation."
        )

        verdict = await llm.vision_chat(question, image_bytes, system=system)
        logger.info("executor: vision QA for %s → %s", image_path.name, verdict[:120])
        return verdict.strip()

    except Exception as exc:
        logger.warning("executor: vision QA failed for %s — %s", image_path.name, exc)
        return f"Vision QA unavailable: {exc}"


def _send_to_slack(
    image_path: Path,
    *,
    channel_id: str,
    thread_ts: str,
    caption: str = "",
) -> None:
    """Upload *image_path* to the Slack channel using the existing slack_tools."""
    from src.tools.slack_tools import slack_send_image, set_slack_channel_context

    if channel_id:
        set_slack_channel_context(channel_id, thread_ts)

    # slack_send_image is a @tool but is fully callable as a plain function.
    result_raw = slack_send_image(
        file_path=str(image_path),
        initial_comment=caption or image_path.name,
    )
    logger.info("executor: Slack upload result for %s → %s", image_path.name, result_raw)


# ---------------------------------------------------------------------------
# Shared post-processing helper
# ---------------------------------------------------------------------------

async def _process_completed_job(
    history: dict,
    prompt_id: str,
    brainbriefing: dict,
    *,
    slack_channel_id: str,
    slack_thread_ts: str,
    verbose: bool,
    collected_paths: list[str] | None,
    label: str = "",
) -> AsyncGenerator[str, None]:
    """Download outputs, run Vision QA, and post to Slack for one finished job.

    Yields one-line status strings.  ``label`` is an optional prefix like
    ``"[2/5] "`` used in batch runs so the user knows which iteration each
    message belongs to.
    """
    pfx = label  # e.g. "[2/5] " or ""

    output_files = _extract_output_files(history)
    if not output_files:
        yield f"{pfx}⚠️ No output files found in ComfyUI history."
        logger.warning("executor: no output files in history for prompt_id=%s", prompt_id)
        return

    saved_paths: list[Path] = []
    for item in output_files:
        filename = item.get("filename", "")
        subfolder = item.get("subfolder", "")
        file_type = item.get("type", "output")
        if not filename:
            continue
        try:
            dest = _download_output(filename, subfolder, file_type)
            saved_paths.append(dest)
            yield f"{pfx}💾 Saved `{filename}` → `{dest}`"
        except Exception as exc:
            yield f"{pfx}⚠️ Could not download `{filename}`: {exc}"
            logger.warning("executor: download failed for %s — %s", filename, exc)

    if not saved_paths:
        yield f"{pfx}❌ All output downloads failed."
        return

    # Vision QA
    yield f"{pfx}🔍 Running Vision QA with Ollama…"
    for path in saved_paths:
        verdict = await _vision_qa(path, brainbriefing)
        yield f"{pfx}🔍 QA `{path.name}` → {verdict}"

    # Post to Slack
    if slack_channel_id or os.environ.get("SLACK_BOT_TOKEN"):
        for path in saved_paths:
            try:
                size_bytes = path.stat().st_size
                target_path = path
                if size_bytes > 5 * 1024 * 1024:
                    yield f"{pfx}⚠️ `{path.name}` is {size_bytes / 1024 / 1024:.1f} MB — downsizing…"
                    target_path = _downsize_for_slack(path)
                    if target_path is None:
                        yield f"{pfx}⚠️ Downsize failed for `{path.name}` — skipping Slack upload."
                        continue
                _send_to_slack(
                    target_path,
                    channel_id=slack_channel_id,
                    thread_ts=slack_thread_ts,
                )
                yield f"{pfx}📤 Sent `{path.name}` to Slack."
            except Exception as exc:
                yield f"{pfx}⚠️ Slack upload failed for `{path.name}`: {exc}"
                logger.warning("executor: slack upload failed for %s — %s", path.name, exc)
    else:
        yield f"{pfx}ℹ️ No Slack token configured — skipping Slack upload."

    output_summary = ", ".join(f"`{p.name}`" for p in saved_paths)
    yield f"{pfx}✅ Done. Outputs: {output_summary}"
    if verbose:
        print(f"[executor] {pfx}Finished. Outputs: {[str(p) for p in saved_paths]}")

    if collected_paths is not None:
        collected_paths.extend(str(p) for p in saved_paths)


# ---------------------------------------------------------------------------
# Public executor — single workflow
# ---------------------------------------------------------------------------

async def execute_workflow(
    workflow_path: str,
    brainbriefing_json: str,
    *,
    slack_channel_id: str = "",
    slack_thread_ts: str = "",
    verbose: bool = True,
    collected_paths: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Submit the validated workflow, poll ComfyUI, QA outputs, post to Slack.

    This is an ``AsyncGenerator[str, None]`` — each yielded string is a one-line
    status update that the pipeline can forward to Slack as a streaming event.

    Args:
        workflow_path:      Absolute path to the validated workflow JSON.
        brainbriefing_json: The Researcher's brainbriefing as a JSON string,
                            used to compare the output against the original brief.
        slack_channel_id:   Slack channel to post results to (empty = no Slack).
        slack_thread_ts:    Slack thread timestamp for replies (empty = new thread).
        verbose:            Log progress to stdout when True.
    """
    from src.utils.comfyui_poller import poll_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    # ── 1. Submit ──────────────────────────────────────────────────────────
    yield "🚀 Submitting workflow to ComfyUI…"
    try:
        prompt_id = _submit_workflow(workflow_path)
    except Exception as exc:
        error_msg = f"❌ ComfyUI submission failed: {exc}"
        logger.error("executor: %s", error_msg)
        yield error_msg
        return

    yield f"✅ Queued · prompt_id=`{prompt_id}` — waiting for completion…"
    if verbose:
        print(f"[executor] Queued prompt_id={prompt_id}")

    # ── 2. Poll ────────────────────────────────────────────────────────────
    history = await poll_comfyui_job(prompt_id)

    if "error" in history:
        error_msg = f"❌ ComfyUI execution error: {history['error']}"
        logger.error("executor: %s", error_msg)
        yield error_msg
        return

    yield "✅ ComfyUI execution complete — collecting outputs…"

    # ── 3-5. Download, QA, Slack ───────────────────────────────────────────
    async for line in _process_completed_job(
        history,
        prompt_id,
        brainbriefing,
        slack_channel_id=slack_channel_id,
        slack_thread_ts=slack_thread_ts,
        verbose=verbose,
        collected_paths=collected_paths,
    ):
        yield line


# ---------------------------------------------------------------------------
# Public executor — batch (submit-all-then-poll)
# ---------------------------------------------------------------------------

async def execute_workflows_batch(
    workflow_paths: list[str],
    brainbriefing_json: str,
    *,
    slack_channel_id: str = "",
    slack_thread_ts: str = "",
    verbose: bool = True,
    collected_paths: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Submit ALL workflows to ComfyUI first, then poll + process each in order.

    This avoids the submit→wait→submit→wait pattern: ComfyUI receives the full
    batch immediately and can start executing jobs while the client is still
    submitting remaining ones.  Polling then starts *after* the last submission,
    so earlier jobs are frequently already done by the time we reach them.

    Args:
        workflow_paths:     Ordered list of absolute workflow JSON file paths.
        brainbriefing_json: Researcher brainbriefing (for Vision QA).
        slack_channel_id:   Slack channel (empty = no Slack).
        slack_thread_ts:    Slack thread ts for replies.
        verbose:            Log progress to stdout when True.
    """
    from src.utils.comfyui_poller import poll_comfyui_job

    try:
        brainbriefing: dict = json.loads(brainbriefing_json)
    except Exception:
        brainbriefing = {}

    total = len(workflow_paths)

    # ── Phase 1: submit all ────────────────────────────────────────────────
    queued: list[tuple[str, str]] = []  # [(prompt_id, workflow_path), ...]
    for idx, wf_path in enumerate(workflow_paths, 1):
        yield f"🚀 Queuing iteration {idx}/{total}…"
        try:
            prompt_id = _submit_workflow(wf_path)
            queued.append((prompt_id, wf_path))
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
        f"waiting for ComfyUI to finish…"
    )

    # ── Phase 2: poll + process each in submission order ───────────────────
    for idx, (prompt_id, _wf_path) in enumerate(queued, 1):
        label = f"[{idx}/{len(queued)}] "
        yield f"{label}⏳ Waiting for ComfyUI (prompt_id=`{prompt_id}`)…"
        if verbose:
            print(f"[executor] Batch polling {idx}/{len(queued)} prompt_id={prompt_id}")

        history = await poll_comfyui_job(prompt_id)

        if "error" in history:
            error_msg = f"{label}❌ ComfyUI execution error: {history['error']}"
            logger.error("executor: %s", error_msg)
            yield error_msg
            continue  # move on to the next iteration, don't abort the whole batch

        yield f"{label}✅ Complete — collecting outputs…"

        async for line in _process_completed_job(
            history,
            prompt_id,
            brainbriefing,
            slack_channel_id=slack_channel_id,
            slack_thread_ts=slack_thread_ts,
            verbose=verbose,
            collected_paths=collected_paths,
            label=label,
        ):
            yield line


def _downsize_for_slack(source: Path) -> Path | None:
    """Downsize *source* image to < 5 MB for Slack using Pillow.

    Saves the result next to the original with a ``_slack`` suffix.
    Returns the downsized path, or ``None`` on failure.
    """
    try:
        from PIL import Image
        import io

        dest = source.with_stem(source.stem + "_slack")
        img = Image.open(source)
        long_edge = max(img.width, img.height)
        target_long_edge = 2048

        if long_edge > target_long_edge:
            ratio = target_long_edge / long_edge
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

        if img.mode == "RGBA":
            img = img.convert("RGB")

        buf = io.BytesIO()
        for quality in range(90, 19, -10):
            buf.seek(0)
            buf.truncate()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= 5 * 1024 * 1024:
                break

        dest.write_bytes(buf.getvalue())
        logger.info("executor: downsized %s → %s (%d bytes)", source.name, dest.name, buf.tell())
        return dest
    except Exception as exc:
        logger.warning("executor: downsize failed for %s — %s", source.name, exc)
        return None
