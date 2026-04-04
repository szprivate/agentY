"""
agentY – Conversation-history summarisation.

``summarize_conversation`` compresses a full Strands Agent message list into a
compact text summary so that only the summary—not the full conversation—is
forwarded to the next agent call, drastically reducing token baggage.

Typical usage
-------------
>>> text_summary = await summarize_conversation(agent.messages)
"""

from __future__ import annotations

import json
import logging

from .llm_functions import LLMFunctions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation-history summarisation
# ---------------------------------------------------------------------------

_SUMMARISE_SYSTEM = """\
You are a conversation summariser for an AI ComfyUI production agent.

You will receive the full message history of an agent conversation (user
requests, assistant actions, tool calls, tool results, etc.).

Your task: produce a CONCISE summary that captures ONLY the information
needed for the agent to continue working effectively in a follow-up call.

Include:
- What the user originally requested (task description / prompt / intent).
- Which workflow template was used and its key parameters (model, resolution,
  seed, steps, cfg, prompt, lora, controlnet, etc.).
- What the agent did (loaded template, patched params, submitted to ComfyUI, etc.).
- Final output file paths / filenames.
- Any errors or issues that occurred.
- The current state: success, partial, or error.

Exclude:
- Raw JSON blobs, node IDs, internal wiring details.
- Full tool-call payloads or verbose API responses.
- Conversational filler, confirmations, or agent reasoning traces.
- Redundant information already covered elsewhere in the summary.

Respond with ONLY the summary text — no markdown fences, no preamble."""

# Hard cap on raw conversation text fed to the summariser to avoid
# blowing up the context window of the small Qwen model.
_MAX_CONVERSATION_CHARS = 24_000


def _flatten_messages(messages: list[dict]) -> str:
    """Convert a Strands message list into a readable text block for the summariser.

    Strands messages use a ``content`` field that may be a plain string or a
    list of typed content blocks (``{"text": ...}``, ``{"toolUse": ...}``,
    ``{"toolResult": ...}``).  This helper normalises both formats into a
    simple ``role: text`` transcript, truncated to ``_MAX_CONVERSATION_CHARS``
    so it fits comfortably in the summariser's context window.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue

        if isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    lines.append(f"[{role}] {block}")
                elif isinstance(block, dict):
                    if "text" in block:
                        lines.append(f"[{role}] {block['text']}")
                    elif "toolUse" in block:
                        tu = block["toolUse"]
                        name = tu.get("name", "?")
                        # Compact representation — skip bulky input payloads
                        lines.append(f"[{role}:tool_call] {name}(…)")
                    elif "toolResult" in block:
                        tr = block["toolResult"]
                        status = tr.get("status", "?")
                        # Include first 300 chars of the result content for context
                        result_content = json.dumps(tr.get("content", ""), ensure_ascii=False)
                        if len(result_content) > 300:
                            result_content = result_content[:300] + "…"
                        lines.append(f"[{role}:tool_result:{status}] {result_content}")

    text = "\n".join(lines)
    if len(text) > _MAX_CONVERSATION_CHARS:
        text = text[-_MAX_CONVERSATION_CHARS:]
        text = "…(earlier messages truncated)…\n" + text
    return text


async def summarize_conversation(messages: list[dict]) -> str:
    """Summarise a Strands Agent message list into a compact text block.

    Uses the cheap Qwen model (configured in ``settings.json`` under
    ``llm.pipeline.llm_functions``) so this adds virtually no cost.

    Parameters
    ----------
    messages:
        The full ``agent.messages`` list from a Strands Agent.

    Returns
    -------
    str
        A concise plain-text summary suitable for injection as the sole
        conversation context in the next agent invocation.
    """
    if not messages:
        return ""

    conversation_text = _flatten_messages(messages)
    if not conversation_text.strip():
        return ""

    llm = LLMFunctions.from_settings()

    llm_messages: list[dict[str, str]] = [
        {"role": "system", "content": _SUMMARISE_SYSTEM},
        {"role": "user",   "content": f"Conversation to summarise:\n\n{conversation_text}"},
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
        # Fallback: return the last assistant text so the agent has *some* context
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:2000]
                if isinstance(content, list):
                    texts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
                    return "\n".join(texts)[:2000]
        return ""
