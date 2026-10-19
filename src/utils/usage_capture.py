"""What a provider says a call cost and used, kept beside Strands' token counts.

Strands' OpenAI client reduces a response's ``usage`` to input, output and cached
tokens and drops the rest. Two things in the rest matter for the bill:

* ``usage.cost`` - OpenRouter (and any gateway that copies it) states the USD
  the call was charged. Anthropic, OpenAI, Gemini and DashScope do not: their
  responses carry token counts only, so for them the cost stays an estimate.
* ``completion_tokens_details.reasoning_tokens`` - reasoning is billed as output.
  OpenAI, DashScope and OpenRouter include it in ``completion_tokens``; Gemini's
  OpenAI-compatible endpoint reports it only through ``total_tokens``.

:class:`UsageCaptureMixin` goes in front of an ``OpenAIModel`` and records, per
model instance, running totals the token hook reads (:func:`snapshot`).
"""
from __future__ import annotations

from typing import Any


def _get(obj: Any, name: str) -> Any:
    """Field of a pydantic usage object or a plain dict; the OpenAI SDK keeps
    fields it does not know (``cost``) in ``model_extra``."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    val = getattr(obj, name, None)
    if val is None:
        val = (getattr(obj, "model_extra", None) or {}).get(name)
    return val


def reported_cost(usage: Any) -> float | None:
    """USD the provider says this call cost, or None when it does not say."""
    raw = _get(usage, "cost")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val >= 0 else None


def reasoning_tokens(usage: Any) -> int:
    try:
        return int(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def thoughts_outside_completion(usage: Any) -> int:
    """Tokens in ``total_tokens`` that neither prompt nor completion counts
    (Gemini's thinking tokens, billed as output)."""
    try:
        total = int(_get(usage, "total_tokens") or 0)
        p = int(_get(usage, "prompt_tokens") or 0)
        c = int(_get(usage, "completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, total - p - c) if total else 0


class UsageCaptureMixin:
    """Put before ``OpenAIModel``. ``thoughts_in_total`` turns on the Gemini rule."""

    thoughts_in_total = False
    reported_cost_total = 0.0   # USD the provider stated, summed over calls
    usage_calls = 0             # calls that returned usage
    cost_calls = 0              # of those, calls that stated a cost
    reasoning_total = 0

    def format_chunk(self, event: dict[str, Any], **kwargs: Any):  # noqa: ANN201
        chunk = super().format_chunk(event, **kwargs)  # type: ignore[misc]
        if event.get("chunk_type") == "metadata":
            try:
                usage = event["data"]
                self.usage_calls += 1
                cost = reported_cost(usage)
                if cost is not None:
                    self.reported_cost_total += cost
                    self.cost_calls += 1
                self.reasoning_total += reasoning_tokens(usage)
                if self.thoughts_in_total and not reasoning_tokens(usage):
                    extra = thoughts_outside_completion(usage)
                    meta = chunk.get("metadata", {}).get("usage")
                    if extra and isinstance(meta, dict):
                        meta["outputTokens"] = int(meta.get("outputTokens", 0)) + extra
                        self.reasoning_total += extra
            except Exception:  # noqa: BLE001 - never lose the call over bookkeeping
                pass
        return chunk


def snapshot(agent: Any) -> tuple[float, int, int]:
    """``(reported_cost_total, usage_calls, cost_calls)`` of *agent*'s model."""
    model = getattr(agent, "model", None)
    return (float(getattr(model, "reported_cost_total", 0.0) or 0.0),
            int(getattr(model, "usage_calls", 0) or 0),
            int(getattr(model, "cost_calls", 0) or 0))


def capturing_openai_model():
    """``OpenAIModel`` with :class:`UsageCaptureMixin` (imported lazily, like the
    Strands client itself)."""
    from strands.models.openai import OpenAIModel  # noqa: PLC0415

    class CapturingOpenAIModel(UsageCaptureMixin, OpenAIModel):
        """Strands' OpenAI client plus the provider's stated cost, when it states one."""

    return CapturingOpenAIModel
