"""Researcher steering handlers for agentY.

Two handlers enforce Researcher-specific guardrails just-in-time:

Handler 3 — JsonOutputEnforcer (rule-based, no LLM)
    After the Researcher produces its final response, checks that the output
    is valid JSON with no surrounding prose or markdown fences. If not,
    returns Guide instructing the agent to output raw JSON only.

Handler 4 — ModelHallucinationGuard (LLM-based, uses LedgerProvider)
    After the Researcher produces its final response, checks whether any model
    paths in the brainbriefing output were returned by get_workflow_template or
    list_models calls in the session ledger. If fabricated paths are found,
    guides the agent to verify via get_models_in_folder / get_model_types or
    mark paths as unverified.
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
    LLMSteeringHandler,
    LedgerProvider,
    ModelSteeringAction,
    Proceed,
    SteeringHandler,
)

logger = logging.getLogger(__name__)

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
# Handler 4: Model path hallucination guard (LLM-based, uses LedgerProvider)
# ---------------------------------------------------------------------------

_HALLUCINATION_GUARD_SYSTEM_PROMPT = """\
You are a guardrail monitor for a ComfyUI workflow resolution agent.

Your job: after the Researcher produces its final brainbriefing JSON, check
whether any model file paths in that JSON were actually returned by
get_workflow_template or get_models_in_folder or get_model_types tool calls
visible in the session ledger.

Decision rules:
1. Parse the final response JSON (the brainbriefing). Look for any string
   values that appear to be file paths — they typically contain directory
   separators (/ or \\) and end in .safetensors, .ckpt, .pt, or similar.
2. Compare each such path against the tool call results recorded in the ledger
   (look in ledger.tool_calls for tool_name in [get_workflow_template,
   get_models_in_folder, get_model_types] and their results).
3. If a model path in the brainbriefing was NOT returned by any of those tool
   calls AND is not a well-known path that appears in prior session messages:
   → decision: "guide"
   → reason: list the suspect paths and instruct the agent to verify them via
     get_models_in_folder / get_model_types, or note them as unverified in the
     brainbriefing.
4. If all model paths can be traced to a tool call result, OR if no file paths
   are present in the brainbriefing:
   → decision: "proceed"

Important: only flag paths that look like filesystem paths to model files.
Do not flag template names, workflow filenames, or image output paths.
"""


class ModelHallucinationGuard(LLMSteeringHandler):
    """LLM-based guard that detects fabricated model file paths in the brainbriefing.

    Uses LedgerProvider (injected by default into LLMSteeringHandler) to access
    the tool call history and compare model paths against returned values.
    """

    name: str = "model_hallucination_guard"

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

        # Delegate to LLM evaluation.
        return await super().steer_after_model(
            agent=agent, message=message, stop_reason=stop_reason, **kwargs
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_researcher_steering_handlers(model=None) -> list:
    """Return the list of steering handler plugins for the Researcher agent.

    Args:
        model: Optional Strands Model instance for the LLM-based handlers.
               Defaults to None, which reuses the Researcher agent's own model
               (local Qwen via Ollama by default — cheapest / free option).

    Returns:
        List of SteeringHandler instances ready to pass as ``plugins=``.
    """
    return [
        JsonOutputEnforcer(),
        ModelHallucinationGuard(
            system_prompt=_HALLUCINATION_GUARD_SYSTEM_PROMPT,
            model=model,  # None → reuse agent's model (local Qwen — free)
            # LedgerProvider is injected by LLMSteeringHandler by default.
        ),
    ]
