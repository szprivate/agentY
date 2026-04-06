"""
agentY – Triage entry point.

Classifies incoming user messages and routes them to the appropriate handler.
Uses a Strands Agent wrapping a small Qwen model via Ollama for fast,
cheap intent classification — no tools, single-turn, stateless.

Typical usage
-------------
>>> from src.agent import create_triage_agent
>>> triage_agent = create_triage_agent()
>>> session = AgentSession(session_id="abc")
>>> result  = await triage(user_message, session, info_context, triage_agent)
>>> handler = route(result)          # "researcher" | "brain" | "answer" | "log_warning"
"""

from __future__ import annotations

import json
import logging
import re

from strands import Agent

from .models import AgentSession, MessageIntent, TriageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of *text*, even if wrapped in a code fence."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def triage(
    user_message: str,
    session: AgentSession,
    info_context: dict,  # noqa: ARG001  — reserved for future use
    agent: Agent,
) -> TriageResult:
    """Classify *user_message* using the Triage agent and return a routing result.

    Parameters
    ----------
    user_message:
        Raw text from the user.
    session:
        Current agent session (used to inject prior context into the message).
    info_context:
        Reserved — no longer used by triage directly.  Answering info_query
        requests is now handled by the dedicated Info agent in the pipeline.
    agent:
        Pre-built Strands Triage agent (created by ``create_triage_agent()``).
        Passed in so the model-availability check only runs once at startup.

    Returns
    -------
    TriageResult
        ``response`` is always ``None``; the pipeline delegates info queries
        to the Info agent.
    """
    # Build a compact session context prefix so the model can distinguish
    # follow-up intents (param_tweak / chain / feedback) from new_request.
    session_hint = ""
    if session.chat_summaries:
        last = session.chat_summaries[-1]
        session_hint = (
            f"[SESSION CONTEXT: last_workflow='{last.workflow_name}', "
            f"status='{last.status}', follow_up_count={session.follow_up_count}]\n\n"
        )

    classify_input = f"{session_hint}{user_message}"

    # Call the triage agent — returns the full response string.
    raw: str = str(agent(classify_input))

    # Reset conversation history so prior exchanges never bleed into the next call.
    try:
        agent.conversation_manager.messages.clear()
    except AttributeError:
        try:
            agent.conversation_manager._messages.clear()  # noqa: SLF001
        except AttributeError:
            pass  # Conversation accumulation is non-critical for a tiny classifier.

    intent     = MessageIntent.new_request
    confidence = 0.0
    try:
        json_str = _extract_json(raw) or raw
        parsed   = json.loads(json_str)
        intent   = MessageIntent(parsed["intent"])
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
        One of ``"researcher"`` | ``"brain"`` | ``"answer"`` | ``"log_warning"``.
    """
    if result.confidence < 0.6:
        return "log_warning"

    match result.intent:
        case MessageIntent.info_query:
            return "answer"
        case MessageIntent.param_tweak | MessageIntent.chain | MessageIntent.feedback:
            return "brain"
        case MessageIntent.new_request:
            return "researcher"
        case _:
            return "researcher"
