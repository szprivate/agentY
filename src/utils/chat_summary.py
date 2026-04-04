"""
agentY – Conversation-history summarisation.

``summarize_conversation`` compresses a full Strands Agent message list into a
compact structured summary so that only the summary—not the full conversation—is
forwarded to the next agent call, drastically reducing token baggage.

Typical usage
-------------
>>> text_summary = await summarize_conversation(agent.messages)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from .llm_functions import LLMFunctions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dedicated message-history file logger → logs/message_history.log
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

_MSG_HISTORY_LOG = str(_PROJECT_ROOT / "logs" / "message_history.log")
os.makedirs(os.path.dirname(_MSG_HISTORY_LOG), exist_ok=True)

_history_logger = logging.getLogger("agentY.message_history")
_history_logger.setLevel(logging.DEBUG)
if not _history_logger.handlers:
    _history_handler = logging.FileHandler(_MSG_HISTORY_LOG, encoding="utf-8")
    _history_handler.setLevel(logging.DEBUG)
    _history_handler.setFormatter(logging.Formatter("%(message)s"))
    _history_logger.addHandler(_history_handler)
    _history_logger.propagate = False


def _log_message_history(messages: list[dict]) -> None:
    """Append the full message list as JSON to logs/message_history.log."""
    sep = "=" * 80
    _history_logger.debug(
        "%s\n[MESSAGE HISTORY — %d message(s)]\n%s\n%s\n%s",
        sep,
        len(messages),
        sep,
        json.dumps(messages, indent=2, ensure_ascii=False),
        sep,
    )


# ---------------------------------------------------------------------------
# Conversation-history summarisation
# ---------------------------------------------------------------------------

_SUMMARISE_SYSTEM = """\
You are a conversation summariser for an AI ComfyUI production agent.

You will receive the full message history of an agent conversation (user
requests, assistant actions, tool calls, tool results, etc.), plus a block
of already-extracted metadata (authoritative — do not change those values).

Your task: output EXACTLY seven labeled lines, in this order:

TASK: <3–8 word production summary, e.g. "image edit, then upscale" | "image generation" | "video i2v" | "downsize, image edit, upscale">
TEMPLATE: <exact workflow template name from the conversation, or "unknown">
WORKFLOW_FILE: <value from extracted metadata — copy verbatim, do NOT change>
INPUT_PATHS: <comma-separated absolute paths of input images/videos from the conversation, or "none">
OUTPUT_PATHS: <value from extracted metadata — copy verbatim, do NOT change>
STATUS: <success | partial | error>
ERRORS: <one-sentence error description, or "none">

Rules:
- Output ONLY the seven labeled lines — no preamble, no markdown, no blank lines between them.
- TASK: extremely brief (≤8 words). Describe the production pipeline, not internal steps.
- TEMPLATE: copy the exact name from a get_workflow_template call in the conversation.
- WORKFLOW_FILE and OUTPUT_PATHS: always copy verbatim from the extracted metadata block.
- INPUT_PATHS: exact file paths from upload_image / LoadImage calls in the conversation.
- STATUS: success if ComfyUI job completed and result posted to Slack; partial if job ran but QA/post failed; error otherwise.
- ERRORS: one sentence max, or "none"."""

# Hard cap on raw conversation text fed to the summariser to avoid
# blowing up the context window of the small Qwen model.
_MAX_CONVERSATION_CHARS = 24_000

# Recognised output-sending tool names whose file_path / save_to inputs are
# treated as generated output paths.
_OUTPUT_TOOLS: frozenset[str] = frozenset({
    "slack_send_image", "slack_send_video", "view_image",
    "slack_send_file", "slack_send_json",
})


def _flatten_messages(messages: list[dict]) -> str:
    """Convert a Strands message list into a readable text block for the summariser.

    Strands messages use a ``content`` field that may be a plain string or a
    list of typed content blocks (``{"text": ...}``, ``{"toolUse": ...}``,
    ``{"toolResult": ...}``).  This helper normalises both formats into a
    simple ``role: text`` transcript, truncated to ``_MAX_CONVERSATION_CHARS``
    so it fits comfortably in the summariser's context window.

    - toolUse blocks include the tool name and key path / name parameters.
    - toolResult blocks extract the inner text content (not raw JSON) and are
      capped at 400 chars each to avoid blowing the context window.
    """
    # Key input parameters to surface for each tool call
    _KEY_PARAMS = ("workflow_path", "file_path", "save_to", "filename", "path", "name")

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            # Truncate long text messages (brainbriefing JSON etc.)
            text = content if len(content) <= 600 else content[:600] + "…"
            lines.append(f"[{role}] {text}")
            continue

        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    lines.append(f"[{role}] {block}")
                elif isinstance(block, dict):
                    if "text" in block:
                        text = block["text"]
                        if len(text) > 600:
                            text = text[:600] + "…"
                        lines.append(f"[{role}] {text}")
                    elif "toolUse" in block:
                        tu = block["toolUse"]
                        name = tu.get("name", "?")
                        inp = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}
                        key_parts: list[str] = []
                        for key in _KEY_PARAMS:
                            val = inp.get(key)
                            if isinstance(val, str) and len(val) < 200:
                                key_parts.append(f"{key}={val!r}")
                        param_str = ", ".join(key_parts[:3]) if key_parts else "…"
                        lines.append(f"[{role}:tool_call] {name}({param_str})")
                    elif "toolResult" in block:
                        tr = block["toolResult"]
                        status = tr.get("status", "?")
                        raw = tr.get("content", "")
                        if isinstance(raw, list):
                            text_parts = [
                                item.get("text", "")
                                for item in raw
                                if isinstance(item, dict) and "text" in item
                            ]
                            result_text = " ".join(text_parts)
                        else:
                            result_text = json.dumps(raw, ensure_ascii=False)
                        if len(result_text) > 400:
                            result_text = result_text[:400] + "…"
                        lines.append(f"[{role}:tool_result:{status}] {result_text}")

    text = "\n".join(lines)
    if len(text) > _MAX_CONVERSATION_CHARS:
        text = text[-_MAX_CONVERSATION_CHARS:]
        text = "…(earlier messages truncated)…\n" + text
    return text


# ---------------------------------------------------------------------------
# Path extraction helpers
# ---------------------------------------------------------------------------

def _extract_paths_from_messages(messages: list[dict]) -> dict:
    """Scan raw Strands messages and extract key production paths.

    Returns a dict with keys:
    - ``workflow_path``  – last patched workflow file path (str | None)
    - ``template_name``  – workflow template name (str | None)
    - ``output_paths``   – ordered list of generated output file paths
    - ``input_paths``    – ordered list of input image/video file paths
    """
    workflow_path: str | None = None
    template_name: str | None = None
    output_paths: list[str] = []
    input_paths: list[str] = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # ── toolUse blocks ─────────────────────────────────────────
            if "toolUse" in block:
                tu = block["toolUse"]
                tool_name: str = tu.get("name", "")
                inp: dict = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}

                if tool_name == "patch_workflow":
                    wp = inp.get("workflow_path")
                    if isinstance(wp, str) and wp:
                        workflow_path = wp  # keep last (most recent) patch target

                elif tool_name == "get_workflow_template":
                    tn = inp.get("name") or inp.get("template_name")
                    if isinstance(tn, str) and tn and not template_name:
                        template_name = tn

                elif tool_name in _OUTPUT_TOOLS:
                    fp = inp.get("file_path") or inp.get("save_to") or inp.get("filename")
                    if isinstance(fp, str) and fp and fp not in output_paths:
                        output_paths.append(fp)

                elif tool_name == "upload_image":
                    fp = inp.get("image_path") or inp.get("filename")
                    if isinstance(fp, str) and fp and fp not in input_paths:
                        input_paths.append(fp)

            # ── toolResult blocks (fallback: mine workflow_path / template) ──
            elif "toolResult" in block:
                tr = block["toolResult"]
                raw = tr.get("content", [])
                items = raw if isinstance(raw, list) else []
                for item in items:
                    if not (isinstance(item, dict) and "text" in item):
                        continue
                    try:
                        data = json.loads(item["text"])
                        if not isinstance(data, dict):
                            continue
                        if not workflow_path:
                            wp = data.get("workflow_path")
                            if isinstance(wp, str) and wp:
                                workflow_path = wp
                        if not template_name:
                            tn = data.get("template_name") or data.get("name")
                            if isinstance(tn, str) and tn:
                                template_name = tn
                    except (json.JSONDecodeError, ValueError):
                        # Fall back to regex scan for paths inside plain text
                        text = item.get("text", "")
                        if not workflow_path:
                            m = re.search(r'"workflow_path"\s*:\s*"([^"]+)"', text)
                            if m:
                                workflow_path = m.group(1)

    return {
        "workflow_path": workflow_path,
        "template_name": template_name,
        "output_paths": output_paths,
        "input_paths": input_paths,
    }


def _archive_workflow(workflow_path: str | None, template_name: str | None) -> str | None:
    """Copy the final patched workflow JSON to ``./output/_workflows/`` for archival.

    Returns the destination path, or ``None`` if the source file is missing or
    ``workflow_path`` is not provided.
    """
    if not workflow_path:
        return None
    src = Path(workflow_path)
    if not src.exists():
        logger.warning("_archive_workflow: source not found: %s", workflow_path)
        return None

    archive_dir = _PROJECT_ROOT / "output" / "_workflows"
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-]", "_", template_name or "workflow")[:50]
    dest = archive_dir / f"{ts}_{safe_name}.json"

    try:
        shutil.copy2(src, dest)
        logger.info("Workflow archived: %s → %s", src, dest)
        return str(dest)
    except OSError as exc:
        logger.warning("_archive_workflow: copy failed (%s)", exc)
        return None


async def summarize_conversation(
    messages: list[dict],
    extra_output_paths: list[str] | None = None,
) -> str:
    """Summarise a Strands Agent message list into a compact structured block.

    Uses the cheap Qwen model (configured in ``settings.json`` under
    ``llm.pipeline.llm_functions``) so this adds virtually no cost.

    The summary is a seven-line labelled block:

    .. code-block:: text

        TASK: image edit, then upscale
        TEMPLATE: image_editing_for_still_images_using_references.model_qwen
        WORKFLOW_FILE: ./output/_workflows/20260404_123456_image_editing.json
        INPUT_PATHS: /path/to/input.jpg
        OUTPUT_PATHS: ./output/result.png
        STATUS: success
        ERRORS: none

    Parameters
    ----------
    messages:
        The full ``agent.messages`` list from a Strands Agent.

    Returns
    -------
    str
        A concise structured summary suitable for injection as the sole
        conversation context in the next agent invocation.
    """
    if not messages:
        return ""

    _log_message_history(messages)

    # ── Programmatic extraction (deterministic, no LLM needed) ───────────
    extracted = _extract_paths_from_messages(messages)
    workflow_file = _archive_workflow(
        extracted["workflow_path"], extracted["template_name"]
    ) or extracted["workflow_path"] or "unknown"

    all_output_paths = list(extracted["output_paths"])
    if extra_output_paths:
        for p in extra_output_paths:
            if p not in all_output_paths:
                all_output_paths.append(p)
    output_paths_str = ", ".join(all_output_paths) if all_output_paths else "none"

    # ── Build hint block injected into the LLM prompt ────────────────────
    hint_lines: list[str] = [
        f"WORKFLOW_FILE (authoritative): {workflow_file}",
        f"OUTPUT_PATHS (authoritative): {output_paths_str}",
    ]
    if extracted["template_name"]:
        hint_lines.append(f"TEMPLATE (authoritative): {extracted['template_name']}")
    if extracted["input_paths"]:
        hint_lines.append(
            f"INPUT_PATHS (authoritative): {', '.join(extracted['input_paths'])}"
        )
    hint_block = "\n".join(hint_lines)

    conversation_text = _flatten_messages(messages)
    if not conversation_text.strip():
        return ""

    llm = LLMFunctions.from_settings()

    llm_messages: list[dict[str, str]] = [
        {"role": "system", "content": _SUMMARISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Extracted metadata (copy WORKFLOW_FILE and OUTPUT_PATHS verbatim):\n"
                f"{hint_block}\n\n"
                f"---\n\n"
                f"Conversation to summarise:\n\n{conversation_text}"
            ),
        },
    ]

    try:
        summary = await llm.chat(llm_messages)
        logger.info(
            "Conversation summarised: %d messages → %d chars",
            len(messages),
            len(summary),
        )
        return summary.strip()
    except Exception as exc:
        logger.warning("Conversation summarisation failed (%s); using fallback", exc)
        # Fallback: return a minimal structured block from extracted metadata
        return (
            f"TASK: unknown\n"
            f"TEMPLATE: {extracted['template_name'] or 'unknown'}\n"
            f"WORKFLOW_FILE: {workflow_file}\n"
            f"INPUT_PATHS: {', '.join(extracted['input_paths']) or 'none'}\n"
            f"OUTPUT_PATHS: {output_paths_str}\n"
            f"STATUS: unknown\n"
            f"ERRORS: summarisation failed — {exc}"
        )
