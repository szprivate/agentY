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
import re
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    path = Path(__file__).parent.parent.parent / "config" / "settings.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _get_ollama_config() -> tuple[str, str]:
    """Return (model_name, host) from settings.json."""
    settings = _load_settings()
    llm = settings.get("llm", {})
    model: str = llm.get("pipeline", {}).get("llm_functions", "qwen3:0.6b")
    host: str = llm.get("ollama", {}).get("host", "http://localhost:11434")
    return model, host


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MessageIntent(str, Enum):
    param_tweak  = "param_tweak"   # adjust a param of the last run
    chain        = "chain"          # pipe last output into a new workflow
    correction   = "correction"     # override a mistake the agent made
    new_request  = "new_request"    # fresh generation request
    info_query   = "info_query"     # question about capabilities / workflows / models


class RunReceipt(BaseModel):
    workflow_name: str
    output_paths: list[str]
    key_params: dict
    user_intent: str
    status: str


class AgentSession(BaseModel):
    session_id: str
    receipts: list[RunReceipt] = Field(default_factory=list)
    current_output_paths: list[str] = Field(default_factory=list)
    follow_up_count: int = 0


class TriageResult(BaseModel):
    intent: MessageIntent
    response: str | None = None   # populated only for info_query
    confidence: float


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
- correction   : User is correcting or overriding a mistake the agent made
- new_request  : User is making a fresh generation request unrelated to prior context
- info_query   : User is asking a factual question about capabilities, workflows, \
or models — NOT requesting generation

Respond with a JSON object only, no markdown, no explanation:
{"intent": "<intent>", "confidence": <float 0.0–1.0>}"""

_ANSWER_SYSTEM = """\
You are a concise assistant for an AI ComfyUI production agent.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say so briefly.
Do not hallucinate workflow names or model names."""


# ---------------------------------------------------------------------------
# Keyword-based context slicing
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "which", "how", "can", "do", "does", "did",
    "i", "you", "it", "in", "of", "for", "to", "and", "or",
    "use", "with", "my", "that", "this", "have", "has",
})

_MAX_WORKFLOWS = 20
_MAX_MODELS    = 8


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _slice_info_context(user_message: str, info_context: dict) -> str:
    """Return a compact, keyword-matched slice of info_context as plain text."""
    keywords = _extract_keywords(user_message)

    # --- workflows ---
    workflows: dict[str, str] = info_context.get("workflows", {})
    matched_wf: dict[str, str] = {
        name: desc
        for name, desc in workflows.items()
        if any(kw in (name + " " + desc).lower() for kw in keywords)
    }
    if not matched_wf and workflows:
        matched_wf = dict(list(workflows.items())[:10])  # fallback: first 10
    matched_wf = dict(list(matched_wf.items())[:_MAX_WORKFLOWS])

    # --- models ---
    models: list = info_context.get("models", [])
    matched_models = [
        m for m in models
        if any(kw in str(m).lower() for kw in keywords)
    ] or models[:_MAX_MODELS]
    matched_models = matched_models[:_MAX_MODELS]

    # --- capabilities ---
    capabilities: str = info_context.get("capabilities", "")

    parts: list[str] = []
    if matched_wf:
        wf_lines = "\n".join(f"- {name}: {desc}" for name, desc in matched_wf.items())
        parts.append(f"## Relevant Workflows\n{wf_lines}")
    if matched_models:
        parts.append("## Models\n" + "\n".join(f"- {m}" for m in matched_models))
    if capabilities:
        parts.append(f"## Capabilities\n{capabilities}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama low-level caller
# ---------------------------------------------------------------------------

async def _chat(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    json_format: bool = False,
) -> str:
    """POST to Ollama /api/chat and return the assistant message content."""
    payload: dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "stream":   False,
    }
    if json_format:
        payload["format"] = "json"

    resp = await client.post(
        f"{host}/api/chat",
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def triage(
    user_message: str,
    session: AgentSession,
    info_context: dict,
) -> TriageResult:
    """Classify *user_message* and — for info queries — answer it.

    Parameters
    ----------
    user_message:
        Raw text from the user.
    session:
        Current agent session (used to give the classifier prior context).
    info_context:
        Injected at startup; shape::

            {
                "workflows": {"name": "description", ...},
                "models": [...],
                "capabilities": "...",
            }

    Returns
    -------
    TriageResult
        ``response`` is ``None`` for every intent except ``info_query``.
    """
    model, host = _get_ollama_config()

    # Build a compact session summary for the classifier (no leaking info_context).
    session_hint = ""
    if session.receipts:
        last = session.receipts[-1]
        session_hint = (
            f"\n\nRecent context: last_workflow='{last.workflow_name}', "
            f"status='{last.status}', follow_up_count={session.follow_up_count}."
        )

    async with httpx.AsyncClient() as client:

        # ── Call 1: intent classification ─────────────────────────────────
        classify_messages = [
            {"role": "system", "content": _CLASSIFY_SYSTEM + session_hint},
            {"role": "user",   "content": user_message},
        ]
        raw = await _chat(client, host, model, classify_messages, json_format=True)

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

        # ── Call 2 (info_query only): answer using sliced context ─────────
        response: str | None = None
        if intent == MessageIntent.info_query:
            context_slice = _slice_info_context(user_message, info_context)
            answer_messages = [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Context:\n{context_slice}\n\nQuestion: {user_message}",
                },
            ]
            response = await _chat(client, host, model, answer_messages)

    return TriageResult(intent=intent, response=response, confidence=confidence)


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
        case MessageIntent.param_tweak | MessageIntent.chain | MessageIntent.correction:
            return "brain"
        case MessageIntent.new_request:
            return "researcher"
        case _:
            return "researcher"
