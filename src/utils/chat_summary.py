"""
agentY – Chat summary generator.

After a workflow completes, ``generate_chat_summary`` makes a single Qwen call
to distil the raw ``params`` dict down to the handful of keys that actually
matter to the user (seed, prompt, resolution, model, …) and wraps everything
into a validated ``ChatSummary``.

Typical usage
-------------
>>> result  = WorkflowResult(workflow_name="flux_t2i", output_paths=["/out/img.png"],
...                           params=raw_params, error=None)
>>> summary = await generate_chat_summary(result, user_intent="Generate a sunset over the sea")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .llm_functions import LLMFunctions
from .models import ChatSummary, WorkflowResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a parameter extractor for an AI ComfyUI production agent.

You will receive a raw params dict that may contain hundreds of internal keys
(node IDs, class_type fields, widget_values arrays, etc.).

Your task: return ONLY the keys that a non-technical user would care about.
Keep: seed, prompt (or positive/negative), resolution (width/height), model \
(checkpoint, lora, controlnet), steps, cfg, denoise, strength, style, \
sampler, scheduler, and any obviously human-readable creative parameters.
Drop: node IDs, class_type, _meta, inputs/outputs wiring, file paths that are \
internal temporary files, and any numeric IDs.

Respond with a single JSON object — no markdown, no explanation:
{"key_params": {"<param>": <value>, ...}}

If no user-relevant params can be identified, return {"key_params": {}}."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_chat_summary(
    result: WorkflowResult,
    user_intent: str,
) -> ChatSummary:
    """Generate a ``ChatSummary`` for a completed workflow run.

    Makes a single Qwen call to extract user-relevant params from
    ``result.params``; all other fields are derived directly.

    Parameters
    ----------
    result:
        The raw outcome of a workflow execution.
    user_intent:
        One-sentence summary of what the user asked for (passed through as-is,
        no LLM needed).

    Returns
    -------
    ChatSummary
        Validated summary.  ``status`` is ``"success"`` when ``result.error``
        is ``None``, otherwise ``"error"``.
    """
    llm = LLMFunctions.from_settings()

    params_json = json.dumps(result.params, ensure_ascii=False, indent=None)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Workflow: {result.workflow_name}\n\n"
                f"Raw params:\n{params_json}"
            ),
        },
    ]

    key_params: dict[str, Any] = {}
    raw = await llm.chat(messages, json_format=True)

    try:
        parsed = json.loads(raw)
        key_params = parsed.get("key_params", {})
        if not isinstance(key_params, dict):
            raise ValueError(f"key_params is not a dict: {type(key_params)}")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Chat summary param extraction failed (%s); raw=%r — using empty key_params",
            exc,
            raw,
        )

    return ChatSummary(
        workflow_name=result.workflow_name,
        output_paths=result.output_paths,
        key_params=key_params,
        user_intent=user_intent,
        status="success" if result.error is None else "error",
    )
