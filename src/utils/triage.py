"""
agentY – Triage entry point.

Classifies incoming user messages and routes them to the appropriate handler.
Uses a small Qwen model (``llm.pipeline.llm_functions``) via Ollama for fast,
cheap intent classification.

Typical usage
-------------
>>> session = AgentSession(session_id="abc")
>>> result  = await triage(user_message, session, info_context)
>>> handler = route(result)          # "researcher" | "brain" | "answer" | "log_warning"
"""

from __future__ import annotations

import json
import logging

from .llm_functions import LLMFunctions
from .models import AgentSession, MessageIntent, TriageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a message intent classifier for an AI image/video generation assistant.

Classify the user message into exactly one of these intents:
- param_tweak  : User wants to adjust a parameter of the last run \
(e.g. change style, resolution, strength, seed)
- chain        : User wants to pipe the last output into a new workflow \
(e.g. "now upscale it", "turn it into a video")
- feedback     : User is providing qualitative feedback or a correction on the generated output \
(e.g. "the face looks wrong", "colors are too saturated", "make it more dramatic", \
"the lighting is off", "that's not what I asked for") — they evaluate the result and want changes
- new_request  : User is making a fresh generation request unrelated to prior context
- info_query   : User is asking a factual question about capabilities, workflows, \
or models — NOT requesting generation
- restart      : User wants to restart or reset the agent \
(e.g. "restart", "reboot", "reset the agent", "start over", "restart the bot")

Respond with a JSON object only, no markdown, no explanation:
{"intent": "<intent>", "confidence": <float 0.0–1.0>}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def triage(
    user_message: str,
    session: AgentSession,
    info_context: dict,  # noqa: ARG001  — reserved for future use
) -> TriageResult:
    """Classify *user_message* and return a routing result.

    Parameters
    ----------
    user_message:
        Raw text from the user.
    session:
        Current agent session (used to give the classifier prior context).
    info_context:
        Reserved — no longer used by triage directly.  Answering info_query
        requests is now handled by the dedicated Info agent in the pipeline.

    Returns
    -------
    TriageResult
        ``response`` is always ``None``; the pipeline delegates info queries
        to the Info agent.
    """
    llm = LLMFunctions.from_settings()

    # Build a compact session summary for the classifier (no leaking info_context).
    session_hint = ""
    if session.chat_summaries:
        last = session.chat_summaries[-1]
        session_hint = (
            f"\n\nRecent context: last_workflow='{last.workflow_name}', "
            f"status='{last.status}', follow_up_count={session.follow_up_count}."
        )

    # ── Call 1: intent classification ─────────────────────────────────────
    classify_messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM + session_hint},
        {"role": "user",   "content": user_message},
    ]
    raw = await llm.chat(classify_messages, json_format=True)

    intent     = MessageIntent.new_request
    confidence = 0.0
    try:
        parsed     = json.loads(raw)
        intent     = MessageIntent(parsed["intent"])
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Intent parse failed (%s); raw=%r — defaulting to new_request", exc, raw)

    # Confidence gate
    if confidence < 0.6:
        logger.warning(
            "Low-confidence classification (%.2f) for %r — defaulting to new_request",
            confidence,
            user_message,
        )
        return TriageResult(
            intent=MessageIntent.new_request,
            response=None,
            confidence=confidence,
        )

    return TriageResult(intent=intent, response=None, confidence=confidence)


def route(result: TriageResult) -> str:
    """Map a *TriageResult* to a handler name.

    Returns
    -------
    str
        One of ``"researcher"`` | ``"brain"`` | ``"answer"`` | ``"restart"`` | ``"log_warning"``.
    """
    if result.confidence < 0.6:
        return "log_warning"

    match result.intent:
        case MessageIntent.info_query:
            return "answer"
        case MessageIntent.restart:
            return "restart"
        case MessageIntent.param_tweak | MessageIntent.chain | MessageIntent.feedback:
            return "brain"
        case MessageIntent.new_request:
            return "researcher"
        case _:
            return "researcher"
