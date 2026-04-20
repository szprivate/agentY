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
    user_message: str | list,
    session: AgentSession,
    info_context: dict,  # noqa: ARG001  — reserved for future use
    agent: Agent,
) -> TriageResult:
    """Classify *user_message* using the Triage agent and return a routing result.

    Parameters
    ----------
    user_message:
        Raw text from the user, or a multimodal content-block list (str + image
        blocks) when the caller wants the triage model to see attached images.
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
    _img_hint = (
        f", user_input_images={session.last_user_input_images}"
        if session.last_user_input_images
        else ""
    )
    if session.chat_summaries:
        last = session.chat_summaries[-1]
        session_hint = (
            f"[SESSION CONTEXT: last_workflow='{last.workflow_name}', "
            f"status='{last.status}', follow_up_count={session.follow_up_count}, "
            f"last_agent='{session.last_agent}'{_img_hint}]\n\n"
        )
    elif session.last_user_input_images:
        # No generations yet but user already uploaded images — surface them so
        # triage won't mistakenly classify follow-up requests as needs_image.
        session_hint = f"[SESSION CONTEXT: no_prior_generation=true{_img_hint}]\n\n"

    if isinstance(user_message, list):
        # Multimodal input: prepend the session hint into the first text block.
        classify_input: str | list
        if session_hint:
            blocks = list(user_message)
            first_text_idx = next((i for i, b in enumerate(blocks) if "text" in b), None)
            if first_text_idx is not None:
                blocks[first_text_idx] = {"text": session_hint + blocks[first_text_idx]["text"]}
            else:
                blocks.insert(0, {"text": session_hint})
            classify_input = blocks
        else:
            classify_input = user_message
    else:
        classify_input = f"{session_hint}{user_message}"

    # Call the triage agent — returns the full response string.
    raw: str = str(agent(classify_input))

    # Reset conversation history so prior exchanges never bleed into the next call.
    # agent.messages is the actual list; conversation_manager does NOT own a messages
    # attribute, so the old conversation_manager.messages.clear() was a no-op.
    try:
        agent.messages.clear()
    except Exception:
        pass

    intent     = MessageIntent.new_request
    confidence = 0.0
    try:
        json_str = _extract_json(raw) or raw
        parsed   = json.loads(json_str)
        intent   = MessageIntent(parsed["intent"])
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Intent parse failed (%s); raw=%r — defaulting to new_request", exc, raw)

    # Chain guard — downgrade to new_request when there is no prior session output.
    # This is the authoritative server-side enforcement of the prompt rule:
    # chain is only valid when at least one successful turn already exists in this thread.
    if intent == MessageIntent.chain and not session.chat_summaries:
        logger.info(
            "Downgrading 'chain' → 'new_request': no prior chat_summaries in session."
        )
        intent = MessageIntent.new_request

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

    # When an image is required but missing, carry a user-facing response so the
    # pipeline can return it directly without starting the Researcher or Brain.
    if intent == MessageIntent.needs_image:
        return TriageResult(
            intent=intent,
            response=(
                "It looks like your request requires an input image, but I don't see one attached. "
                "Please share the image you'd like me to work with and I'll get started!"
            ),
            confidence=confidence,
        )

    return TriageResult(intent=intent, response=None, confidence=confidence)


def route(result: TriageResult) -> str:
    """Map a *TriageResult* to a handler name.

    Returns
    -------
    str
        One of ``"researcher"`` | ``"brain"`` | ``"answer"`` | ``"needs_image"`` | ``"log_warning"``.
    """
    if result.confidence < 0.6:
        return "log_warning"

    match result.intent:
        case MessageIntent.info_query:
            return "answer"
        case MessageIntent.needs_image:
            return "needs_image"
        case MessageIntent.param_tweak | MessageIntent.chain | MessageIntent.feedback:
            return "brain"
        case MessageIntent.new_planned_request:
            return "planner"
        case MessageIntent.new_request:
            return "researcher"
        case _:
            return "researcher"
