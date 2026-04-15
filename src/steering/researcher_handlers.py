"""Researcher steering handlers for agentY.

Two handlers enforce Researcher-specific guardrails just-in-time:

Handler 3 — JsonOutputEnforcer (rule-based, no LLM)
    After the Researcher produces its final response, checks that the output
    is valid JSON with no surrounding prose or markdown fences. If not,
    returns Guide instructing the agent to output raw JSON only.

Handler 4 — BriefingHallucinationGuard (rule-based, no LLM, uses LedgerProvider)
    After the Researcher produces its final response, performs two checks:
    a) Extracts model file paths from the brainbriefing JSON and verifies each
       was returned by get_workflow_template, get_models_in_folder, or
       get_model_types tool calls recorded in the session ledger.
    b) Extracts input image filenames from input_images / input_nodes and
       verifies each against file paths used in analyze_image,
       get_image_resolution, or upload_image tool calls, plus any file paths
       visible in the original user message stored in the agent conversation.
    If fabricated/mistyped paths are found, guides the agent to correct them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from strands import Agent
from strands.types.content import Message
from strands.vended_plugins.steering import (
    Guide,
    LedgerAfterToolCall,
    LedgerProvider,
    ModelSteeringAction,
    Proceed,
    SteeringHandler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe LedgerProvider — strips non-JSON-serializable values from tool results
# ---------------------------------------------------------------------------

def _sanitize_for_json(obj: Any, _depth: int = 0) -> Any:
    """Recursively replace non-JSON-serializable values with a safe placeholder.

    LedgerAfterToolCall stores tool result content verbatim. Tools like
    analyze_image may return raw bytes (e.g. base64-decoded image data),
    which crash strands' json_dict validator. This function converts:
      - bytes  → "<binary: N bytes>"
      - any other non-serializable type → str(obj)
    """
    if _depth > 20:  # guard against pathological nesting
        return "<truncated>"
    if isinstance(obj, bytes):
        return f"<binary: {len(obj)} bytes>"
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item, _depth + 1) for item in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


class _SanitizedLedgerAfterToolCall(LedgerAfterToolCall):
    """LedgerAfterToolCall that sanitizes tool result content before storing.

    Inherits the full ledger entry structure but replaces any non-JSON-
    serializable values (bytes, etc.) in the result content with safe strings.
    This prevents crashes when tools like analyze_image return binary data.
    """

    def __call__(self, event: Any, steering_context: Any, **kwargs: Any) -> None:
        # Sanitize the result content before delegating to the parent writer.
        # We patch event.result temporarily to avoid mutating the live object.
        try:
            content = event.result.get("content")
            if content is not None:
                sanitized = _sanitize_for_json(content)
                original_result = event.result
                patched: dict = {**original_result, "content": sanitized}
                event.result = patched
                try:
                    super().__call__(event, steering_context, **kwargs)
                finally:
                    event.result = original_result
                return
        except Exception:
            pass  # Fall back to raw parent call if patching fails.
        super().__call__(event, steering_context, **kwargs)


class _SanitizedLedgerProvider(LedgerProvider):
    """LedgerProvider that uses _SanitizedLedgerAfterToolCall instead of the
    default LedgerAfterToolCall, preventing JSON serialization crashes from
    tools that return binary content (e.g. analyze_image).
    """

    def context_providers(self, **kwargs: Any) -> list:  # type: ignore[override]
        providers = super().context_providers(**kwargs)
        return [
            _SanitizedLedgerAfterToolCall() if isinstance(p, LedgerAfterToolCall) else p
            for p in providers
        ]


# ---------------------------------------------------------------------------
# Handler 3: JSON output enforcer (rule-based)
# ---------------------------------------------------------------------------

_JSON_GUIDE_REASON = (
    "Your response contains non-JSON content (markdown fences, preamble, "
    "trailing prose, or is not valid JSON). "
    "Output the brainbriefing as raw JSON only — no ```json fences, no "
    "explanatory text before or after. Start your response with `{` and "
    "end with `}`."
)

_MD_FENCE_RE = re.compile(r"```", re.MULTILINE)


def _extract_text(message: Message) -> str:
    """Extract all text content from a strands Message object."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


class JsonOutputEnforcer(SteeringHandler):
    """Enforces that the Researcher's final output is bare JSON.

    Only fires when stop_reason is 'end_turn' (i.e. the model is done, not
    mid-tool-call), so it doesn't interfere with intermediate status messages.
    """

    name: str = "json_output_enforcer"

    async def steer_after_model(
        self,
        *,
        agent: Agent,
        message: Message,
        stop_reason: Literal[
            "cancelled", "content_filtered", "end_turn", "guardrail_intervened",
            "interrupt", "max_tokens", "stop_sequence", "tool_use"
        ],
        **kwargs: Any,
    ) -> ModelSteeringAction:
        # Only check final responses, not tool-call turns or interrupted turns.
        if stop_reason != "end_turn":
            return Proceed(reason="not a final response turn")

        text = _extract_text(message).strip()

        # Reject if empty.
        if not text:
            return Proceed(reason="empty response — allow agent to continue")

        # Reject if markdown fences are present.
        if _MD_FENCE_RE.search(text):
            logger.debug("json_output_enforcer: markdown fences found")
            return Guide(reason=_JSON_GUIDE_REASON)

        # Reject if the text doesn't look like JSON at all.
        if not text.startswith("{"):
            logger.debug("json_output_enforcer: response does not start with {")
            return Guide(reason=_JSON_GUIDE_REASON)

        # Try parsing as JSON.
        try:
            json.loads(text)
            logger.debug("json_output_enforcer: valid JSON — proceeding")
            return Proceed(reason="valid JSON output")
        except json.JSONDecodeError as exc:
            logger.debug("json_output_enforcer: invalid JSON — %s", exc)
            return Guide(
                reason=(
                    f"Your response is not valid JSON (parse error: {exc}). "
                    "Output raw JSON only — no prose, no fences. "
                    "Start with `{` and end with `}`."
                )
            )


# ---------------------------------------------------------------------------
# Handler 4: Briefing hallucination guard (rule-based, uses LedgerProvider)
# ---------------------------------------------------------------------------

# File extensions that indicate model weight files.
_MODEL_EXTENSIONS = re.compile(
    r"\.(safetensors|ckpt|pt|pth|bin|onnx|gguf)$", re.IGNORECASE
)

# File extensions that indicate image/video files.
_IMAGE_EXTENSIONS = re.compile(
    r"\.(png|jpe?g|gif|webp|bmp|tiff?|mp4|mov|avi|mkv|webm)$", re.IGNORECASE
)

# Tool names whose results contain authoritative model/file paths.
_PATH_SOURCE_TOOLS = frozenset({
    "get_workflow_template",
    "get_models_in_folder",
    "get_model_types",
})

# Tool names whose *arguments* contain authoritative image file paths.
_IMAGE_ARG_TOOLS = frozenset({
    "analyze_image",
    "get_image_resolution",
    "upload_image",
})


def _extract_model_paths(obj: Any, _depth: int = 0) -> set[str]:
    """Recursively extract strings that look like model file paths."""
    if _depth > 30:
        return set()
    paths: set[str] = set()
    if isinstance(obj, str):
        # Must contain a path separator and end with a model extension.
        if ("/" in obj or "\\" in obj) and _MODEL_EXTENSIONS.search(obj):
            paths.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            paths.update(_extract_model_paths(v, _depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            paths.update(_extract_model_paths(item, _depth + 1))
    return paths


def _extract_ledger_known_strings(ledger: dict) -> set[str]:
    """Collect all string fragments from relevant tool call results in the ledger.

    We flatten every result from ``_PATH_SOURCE_TOOLS`` into a set of strings
    so that a simple substring check can verify each model path.
    """
    known: set[str] = set()
    for call in ledger.get("tool_calls", []):
        if call.get("tool_name") not in _PATH_SOURCE_TOOLS:
            continue
        result = call.get("result")
        if result is None:
            continue
        # Flatten the result into individual strings.
        _collect_strings(result, known)
    return known


def _extract_ledger_image_filenames(ledger: dict) -> set[str]:
    """Collect authoritative image filenames from tool call arguments in the ledger.

    When the Researcher calls ``analyze_image(file_path=...)``,
    ``get_image_resolution(image_path=...)``, or ``upload_image(file_path=...)``,
    the file paths are recorded in the ledger's ``tool_args``.  We extract the
    basename of each to build a set of known-good image filenames.
    """
    filenames: set[str] = set()
    for call in ledger.get("tool_calls", []):
        tool_name = call.get("tool_name", "")
        if tool_name not in _IMAGE_ARG_TOOLS:
            continue
        args = call.get("tool_args") or {}
        for key in ("file_path", "image_path", "image_url"):
            val = args.get(key, "")
            if val and isinstance(val, str):
                # Store both the full path and just the basename.
                filenames.add(val)
                basename = val.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if basename:
                    filenames.add(basename)
    return filenames


def _extract_conversation_filenames(agent: Agent) -> set[str]:
    """Extract image/video filenames from the agent's conversation messages.

    The pipeline injects on-disk file paths into the user message text
    (e.g. ``files/chainlit_uploads/937ade92-…-ec9d.jpg``).  We scan user
    messages for tokens that look like image file paths and collect the
    basenames.
    """
    filenames: set[str] = set()
    try:
        messages = agent.messages
    except Exception:
        return filenames
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
        for text in texts:
            # Find tokens that look like file paths with image extensions.
            for token in re.findall(r'["\']?([^\s"\',]+\.[a-zA-Z0-9]+)["\']?', text):
                if _IMAGE_EXTENSIONS.search(token):
                    filenames.add(token)
                    basename = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    if basename:
                        filenames.add(basename)
    return filenames


def _extract_briefing_image_filenames(briefing: dict) -> set[str]:
    """Extract image filenames from input_images and input_nodes in a brainbriefing."""
    filenames: set[str] = set()
    for entry in briefing.get("input_images", []):
        if isinstance(entry, dict):
            fn = entry.get("filename", "")
            if fn:
                filenames.add(fn)
    for entry in briefing.get("input_nodes", []):
        if isinstance(entry, dict):
            fn = entry.get("filename", "")
            if fn:
                filenames.add(fn)
            path = entry.get("path", "")
            if path:
                filenames.add(path)
                basename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if basename:
                    filenames.add(basename)
    return filenames


def _collect_strings(obj: Any, acc: set[str], _depth: int = 0) -> None:
    """Recursively collect all string values from a nested structure."""
    if _depth > 30:
        return
    if isinstance(obj, str):
        acc.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, acc, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, acc, _depth + 1)


class BriefingHallucinationGuard(SteeringHandler):
    """Rule-based guard that detects fabricated model paths AND mistyped image filenames.

    Uses LedgerProvider (via context_providers) to record tool call history
    during the Researcher session.  On the final ``end_turn``:

    1. **Model paths** — extracts file paths ending in model-weight extensions
       from the brainbriefing and verifies each against strings returned by
       authoritative tools (get_workflow_template, get_models_in_folder,
       get_model_types).
    2. **Image filenames** — extracts ``input_images[].filename`` and
       ``input_nodes[].filename`` / ``.path`` from the brainbriefing and
       verifies each against file paths that appeared in tool call arguments
       (analyze_image, get_image_resolution, upload_image) and in the original
       user message.

    No LLM call is made — pure string matching.
    """

    name: str = "briefing_hallucination_guard"

    async def steer_after_model(
        self,
        *,
        agent: Agent,
        message: Message,
        stop_reason: Literal[
            "cancelled", "content_filtered", "end_turn", "guardrail_intervened",
            "interrupt", "max_tokens", "stop_sequence", "tool_use"
        ],
        **kwargs: Any,
    ) -> ModelSteeringAction:
        # Only check final responses.
        if stop_reason != "end_turn":
            return Proceed(reason="not a final response turn")

        text = _extract_text(message).strip()
        if not text:
            return Proceed(reason="empty response")

        # Parse the brainbriefing JSON.
        try:
            briefing = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return Proceed(reason="response is not JSON — let JsonOutputEnforcer handle it")

        ledger = self.steering_context.data.get("ledger") or {}
        issues: list[str] = []

        # ---- Check 1: model file paths --------------------------------
        model_paths = _extract_model_paths(briefing)
        if model_paths:
            known_strings = _extract_ledger_known_strings(ledger)
            if known_strings:
                suspect_models: list[str] = []
                for path in model_paths:
                    if any(path in s or s in path for s in known_strings if len(s) > 5):
                        continue
                    basename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    if any(basename in s for s in known_strings if len(s) > 5):
                        continue
                    suspect_models.append(path)
                if suspect_models:
                    issues.append(
                        "Model file paths not verified against any "
                        "get_workflow_template / get_models_in_folder / get_model_types "
                        "tool result: " + ", ".join(suspect_models)
                        + ". Verify via get_models_in_folder / get_model_types or correct them."
                    )

        # ---- Check 2: input image filenames ---------------------------
        briefing_images = _extract_briefing_image_filenames(briefing)
        if briefing_images:
            # Build the set of known-good image filenames from tool args + conversation.
            known_images = _extract_ledger_image_filenames(ledger)
            known_images.update(_extract_conversation_filenames(agent))

            if known_images:
                suspect_images: list[str] = []
                for img in briefing_images:
                    basename = img.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    # Exact match on full string or basename.
                    if img in known_images or basename in known_images:
                        continue
                    # Substring match (known path contains the filename or vice-versa).
                    if any(img in k or basename in k for k in known_images if len(k) > 3):
                        continue
                    suspect_images.append(img)
                if suspect_images:
                    issues.append(
                        "Input image filenames not found in any tool call argument or "
                        "user message: " + ", ".join(suspect_images)
                        + ". These may be mistyped. Check the exact filenames from "
                        "the user's attached files or from analyze_image / "
                        "get_image_resolution results and correct them in the brainbriefing."
                    )

        if not issues:
            logger.debug("briefing_hallucination_guard: all paths verified")
            return Proceed(reason="all model paths and image filenames verified")

        logger.debug("briefing_hallucination_guard: issues found: %s", issues)
        return Guide(reason=" | ".join(issues))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_researcher_steering_handlers(model=None) -> list:
    """Return the list of steering handler plugins for the Researcher agent.

    Args:
        model: Unused — retained for API compatibility.  Both handlers are
               now rule-based and make zero LLM calls.

    Returns:
        List of SteeringHandler instances ready to pass as ``plugins=``.
    """
    return [
        JsonOutputEnforcer(),
        BriefingHallucinationGuard(
            # Use _SanitizedLedgerProvider so tools that return binary content
            # (e.g. analyze_image) don't crash the steering context serialiser.
            context_providers=[_SanitizedLedgerProvider()],
        ),
    ]
