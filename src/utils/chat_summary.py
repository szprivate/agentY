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


def _load_config() -> dict:
    """Load config/settings.json."""
    config_path = _PROJECT_ROOT / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as _f:
            return json.loads("".join(ln for ln in _f if not ln.lstrip().startswith("//")))
    return {}

_config = _load_config()
_MSG_HISTORY_LOG = str(_PROJECT_ROOT / _config.get("message_history_log", "./.logs/message_history.log"))
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
TEMPLATE: <value from extracted metadata — copy verbatim, do NOT change>
WORKFLOW_FILE: <value from extracted metadata — copy verbatim, do NOT change>
INPUT_PATHS: <value from extracted metadata — copy verbatim, do NOT change>
INPUT_PATHS_USER_MESSAGE: <value from extracted metadata — copy verbatim, do NOT change>
OUTPUT_PATHS: <value from extracted metadata — copy verbatim, do NOT change>
PATCHES: <value from extracted metadata — copy verbatim, do NOT change>

Rules:
- Output ONLY the seven labeled lines — no preamble, no markdown, no blank lines between them.
- TASK: extremely brief (≤8 words). Describe the production pipeline, not internal steps.
- TEMPLATE, WORKFLOW_FILE, INPUT_PATHS, INPUT_PATHS_USER_MESSAGE, OUTPUT_PATHS, PATCHES: always copy verbatim from the extracted metadata block."""

# Hard cap on raw conversation text fed to the summariser to avoid
# blowing up the context window of the small Qwen model.
_MAX_CONVERSATION_CHARS = 24_000

# Recognised output-sending tool names whose file_path / save_to inputs are
# treated as generated output paths.
_OUTPUT_TOOLS: frozenset[str] = frozenset({
    "view_image",
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

def _stem_without_timestamp(path: str) -> str:
    """Strip a leading ``YYYYMMDD_HHMMSS_`` timestamp from a file stem."""
    stem = Path(path).stem
    m = re.match(r"^\d{8}_\d{6}_(.+)$", stem)
    return m.group(1) if m else stem


def _extract_paths_from_messages(messages: list[dict]) -> dict:
    """Scan raw Strands messages and extract key production paths.

    Returns a dict with keys:
    - ``workflow_path``          – last patched workflow file path (str | None)
    - ``template_name``          – workflow template name, researcher-preferred (str | None)
    - ``output_paths``           – ordered list of generated output file paths
    - ``input_paths``            – ordered list of input image/video file paths
    - ``patch_changes``          – ordered list of human-readable patch descriptions
    """
    workflow_path: str | None = None
    brainbriefing_template: str | None = None   # researcher-selected (highest priority)
    get_template_name: str | None = None         # brain's get_workflow_template call
    prior_summary_template: str | None = None    # from injected prior-round summary
    output_paths: list[str] = []
    input_paths: list[str] = []
    prior_summary_input_paths: list[str] = []   # from injected prior-round summary
    patch_image_values: list[str] = []          # "image" inputs patched, as fallback
    patch_changes: list[str] = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # ── Plain text blocks ──────────────────────────────────────
            if "text" in block:
                text = block["text"]

                # Brainbriefing injected by pipeline: "template": {"name": "..."}
                if not brainbriefing_template:
                    m = re.search(
                        r'"template"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"',
                        text,
                        re.DOTALL,
                    )
                    if m:
                        brainbriefing_template = m.group(1)

                # Prior-round summary injected as first user message:
                # "[CONVERSATION SUMMARY FROM PRIOR ROUND]\n..."
                if "[CONVERSATION SUMMARY FROM PRIOR ROUND]" in text:
                    if not prior_summary_template:
                        m = re.search(r"^TEMPLATE:\s*(.+)$", text, re.MULTILINE)
                        if m:
                            val = m.group(1).strip()
                            if val and val.lower() not in ("unknown", "none", ""):
                                prior_summary_template = val

                    ip_match = re.search(r"^INPUT_PATHS:\s*(.+)$", text, re.MULTILINE)
                    if ip_match:
                        val = ip_match.group(1).strip()
                        if val.lower() not in ("none", "unknown", ""):
                            for p in val.split(","):
                                p = p.strip()
                                if p:
                                    prior_summary_input_paths.append(p)

                    op_match = re.search(r"^OUTPUT_PATHS:\s*(.+)$", text, re.MULTILINE)
                    if op_match:
                        val = op_match.group(1).strip()
                        if val.lower() not in ("none", "unknown", ""):
                            for p in val.split(","):
                                p = p.strip()
                                if p and p not in output_paths:
                                    output_paths.append(p)

            # ── toolUse blocks ─────────────────────────────────────────
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_name: str = tu.get("name", "")
                inp: dict = tu.get("input", {}) if isinstance(tu.get("input"), dict) else {}

                if tool_name in ("patch_workflow", "update_workflow"):
                    wp = inp.get("workflow_path")
                    if isinstance(wp, str) and wp:
                        workflow_path = wp  # keep last (most recent) patch target
                    # Extract human-readable patch descriptions
                    raw_patches = inp.get("patches")
                    if isinstance(raw_patches, str):
                        try:
                            patch_list = json.loads(raw_patches)
                            if isinstance(patch_list, list):
                                for patch in patch_list:
                                    nid = str(patch.get("node_id", "?"))
                                    if "class_type" in patch:
                                        patch_changes.append(
                                            f"Node {nid}: class_type → {patch['class_type']}"
                                        )
                                    elif "widget_values_index" in patch:
                                        idx = patch["widget_values_index"]
                                        val_r = repr(patch.get("value", "?"))
                                        if len(val_r) > 80:
                                            val_r = val_r[:77] + "..."
                                        patch_changes.append(
                                            f"Node {nid}.widget_values[{idx}] → {val_r}"
                                        )
                                    elif "input_name" in patch:
                                        inp_name = patch["input_name"]
                                        val = patch.get("value")
                                        val_r = repr(val)
                                        if len(val_r) > 80:
                                            val_r = val_r[:77] + "..."
                                        patch_changes.append(
                                            f"Node {nid}.{inp_name} → {val_r}"
                                        )
                                        # Capture "image" inputs with string values as
                                        # fallback input-path hints (LoadImage patches)
                                        if (
                                            inp_name == "image"
                                            and isinstance(val, str)
                                            and val
                                        ):
                                            patch_image_values.append(val)
                        except (json.JSONDecodeError, TypeError):
                            pass

                elif tool_name == "get_workflow_template":
                    tn = inp.get("name") or inp.get("template_name")
                    if isinstance(tn, str) and tn and not get_template_name:
                        get_template_name = tn

                elif tool_name in _OUTPUT_TOOLS:
                    fp = inp.get("file_path") or inp.get("save_to") or inp.get("filename")
                    if isinstance(fp, str) and fp and fp not in output_paths:
                        output_paths.append(fp)

                elif tool_name == "upload_image":
                    # The tool uses "file_path"; also accept legacy "image_path"/"filename"
                    fp = inp.get("file_path") or inp.get("image_path") or inp.get("filename")
                    if isinstance(fp, str) and fp and fp not in input_paths:
                        input_paths.append(fp)

            # ── toolResult blocks (fallback: mine workflow_path) ───────
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
                    except (json.JSONDecodeError, ValueError):
                        # Fall back to regex scan for paths inside plain text
                        text = item.get("text", "")
                        if not workflow_path:
                            m = re.search(r'"workflow_path"\s*:\s*"([^"]+)"', text)
                            if m:
                                workflow_path = m.group(1)

    # Priority: researcher brainbriefing > get_workflow_template call > prior summary
    template_name = brainbriefing_template or get_template_name or prior_summary_template

    # Fallback input paths: use prior-summary values, then LoadImage patch values
    if not input_paths:
        if prior_summary_input_paths:
            input_paths = list(prior_summary_input_paths)
        elif patch_image_values:
            input_paths = list(dict.fromkeys(patch_image_values))  # deduplicated

    return {
        "workflow_path": workflow_path,
        "template_name": template_name,
        "output_paths": output_paths,
        "input_paths": input_paths,
        "patch_changes": patch_changes,
    }


def _archive_workflow(workflow_path: str | None, template_name: str | None) -> str | None:
    """Copy the final patched workflow JSON to ``./output_workflows/`` for archival.

    Returns the destination path, or ``None`` if the source file is missing or
    ``workflow_path`` is not provided.
    """
    if not workflow_path:
        return None
    src = Path(workflow_path)
    if not src.exists():
        logger.warning("_archive_workflow: source not found: %s", workflow_path)
        return None

    archive_dir = (_PROJECT_ROOT / _load_config().get("output_workflows_dir", "./output_workflows/")).resolve()
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
    user_message_image_paths: list[str] | None = None,
) -> str:
    """Summarise a Strands Agent message list into a compact structured block.

    Uses the cheap Qwen model (configured in ``settings.json`` under
    ``llm.pipeline.llm_functions``) so this adds virtually no cost.

    The summary is a seven-line labelled block:

    .. code-block:: text

        TASK: image edit, then upscale
        TEMPLATE: image_editing_for_still_images_using_references.model_qwen
        WORKFLOW_FILE: ./output_workflows/20260404_123456_image_editing.json
        INPUT_PATHS: /path/to/input.jpg
        INPUT_PATHS_USER_MESSAGE: /path/to/original_upload.jpg
        OUTPUT_PATHS: ./output/result.png
        PATCHES: Node 6.text → 'a photograph of ...', Node 190.image → 'input.png'

    Parameters
    ----------
    messages:
        The full ``agent.messages`` list from a Strands Agent.
    extra_output_paths:
        Executor-produced file paths not visible in the Brain's message history.
    user_message_image_paths:
        Original file paths of images the user attached to their message
        (from ``AgentSession.last_user_input_images``).  Always captured
        verbatim so the next session can use them as authoritative input paths.

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

    # ── User-message image paths (authoritative, never extracted from tool calls) ─
    user_img_paths_str = (
        ", ".join(user_message_image_paths)
        if user_message_image_paths
        else "none"
    )

    # ── Build hint block injected into the LLM prompt ────────────────────
    patches_str = ", ".join(extracted["patch_changes"]) if extracted["patch_changes"] else "none"
    input_paths_str = ", ".join(extracted["input_paths"]) if extracted["input_paths"] else "none"

    # If no template name was resolved through any source, fall back to
    # "same as WORKFLOW_FILE" so the Brain always has a usable reference.
    template_display = extracted["template_name"]
    if not template_display:
        if extracted["workflow_path"]:
            template_display = f"same as WORKFLOW_FILE ({_stem_without_timestamp(extracted['workflow_path'])})"
        else:
            template_display = "unknown"

    hint_lines: list[str] = [
        f"TEMPLATE (authoritative): {template_display}",
        f"WORKFLOW_FILE (authoritative): {workflow_file}",
        f"INPUT_PATHS (authoritative): {input_paths_str}",
        f"INPUT_PATHS_USER_MESSAGE (authoritative): {user_img_paths_str}",
        f"OUTPUT_PATHS (authoritative): {output_paths_str}",
        f"PATCHES (authoritative): {patches_str}",
    ]
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
                f"Extracted metadata (copy all authoritative fields verbatim):\n"
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
            f"TEMPLATE: {template_display}\n"
            f"WORKFLOW_FILE: {workflow_file}\n"
            f"INPUT_PATHS: {input_paths_str}\n"
            f"INPUT_PATHS_USER_MESSAGE: {user_img_paths_str}\n"
            f"OUTPUT_PATHS: {output_paths_str}\n"
            f"PATCHES: {patches_str}"
        )
