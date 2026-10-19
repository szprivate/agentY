"""Gemini through its OpenAI-compatible endpoint, with thought signatures kept.

Gemini 3 attaches a ``thought_signature`` to every function call it makes and
rejects the next request (HTTP 400, "Function call is missing a thought_signature
in functionCall parts") unless the signature comes back unchanged on that call.
On the OpenAI-compatible endpoint it travels as
``tool_calls[].extra_content.google.thought_signature``. Strands' stock
``OpenAIModel`` reads neither direction, so the signature was lost at the first
tool round trip.

This subclass reads it off the streamed tool call into the toolUse block's
``reasoningSignature`` (the field Strands' event loop already carries through
the message history) and writes it back onto the request's tool call.
"""

from __future__ import annotations

from typing import Any

from strands.models.openai import OpenAIModel

from src.utils.usage_capture import UsageCaptureMixin

# Gemini's documented stand-in for a call that has no real signature (history
# written by another model, or saved before signatures were kept): it makes the
# validator skip the check instead of failing the request.
DUMMY_SIGNATURE = "skip_thought_signature_validator"


def _signature_of(tool_call: Any) -> str | None:
    """The thought signature on a streamed tool-call delta, if it has one."""
    extra = getattr(tool_call, "extra_content", None)
    if extra is None:
        extra = (getattr(tool_call, "model_extra", None) or {}).get("extra_content")
    if isinstance(extra, dict):
        sig = (extra.get("google") or {}).get("thought_signature")
        return sig if isinstance(sig, str) and sig else None
    return None


class GeminiCompatModel(UsageCaptureMixin, OpenAIModel):
    """OpenAIModel that round-trips Gemini 3 thought signatures."""

    # Thinking tokens are billed as output but only show up in total_tokens.
    thoughts_in_total = True

    @classmethod
    def format_request_message_tool_call(cls, tool_use, **kwargs: Any) -> dict[str, Any]:
        call = super().format_request_message_tool_call(tool_use, **kwargs)
        call["extra_content"] = {"google": {
            "thought_signature": tool_use.get("reasoningSignature") or DUMMY_SIGNATURE}}
        return call

    def format_chunk(self, event: dict[str, Any], **kwargs: Any):
        chunk = super().format_chunk(event, **kwargs)
        if event.get("chunk_type") == "content_start" and event.get("data_type") == "tool":
            sig = _signature_of(event["data"])
            if sig:
                chunk["contentBlockStart"]["start"]["toolUse"]["reasoningSignature"] = sig
        return chunk
