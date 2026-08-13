"""Deliver mid-run user messages to the orchestrator at the next tool boundary.

The user types something while a turn is running and hits "send now"; the text
lands in :mod:`src.utils.interject_bus`. This hook is what gets it in front of the
model, using the two fields Strands lets a hook write:

* **normal** — ``AfterToolCallEvent.result``: the tool runs, and the user's words
  are appended to the result the model reads next. Nothing about the agent loop
  changes and the message arrives inside a ``tool_result`` block, which is exactly
  what the provider requires after a ``tool_use`` — a bare user message injected
  there is an HTTP 400 with Anthropic and DashScope alike.
* **urgent** — ``BeforeToolCallEvent.cancel_tool``: the pending call is cancelled
  and the message becomes its (error) result, so the model reads it *instead of*
  doing the thing. Same idiom Strands' own steering plugin uses for ``Guide``.

Either way the wait is bounded by the current tool call, not by the turn: a
canvas batch or a specialist delegation is one long call, and the interjection
lands when it returns.

The wording of both envelopes lives in
``config/system_prompts/orchestrator/interjection*.md`` — prompt text belongs in
a file, not in a string literal here.
"""

from __future__ import annotations

import logging
from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

from agenty_core.utils.progress_signal import push as _push_progress
from src.utils import interject_bus

logger = logging.getLogger("agentY.interject")

# Fallback label if the partial is missing/unreadable. Deliberately a bare marker
# and not guidance: the guidance is the .md file's job, but an interjection must
# never reach the model looking like the tool's own output.
_FALLBACK = "[USER INTERJECTION]"


def _envelope(urgent: bool) -> str:
    """The instruction block that precedes the user's words."""
    try:
        from src.pipeline import _orch_partial  # lazy: pipeline imports agent imports this
        text = _orch_partial("interjection_urgent" if urgent else "interjection")
    except Exception:  # noqa: BLE001
        text = ""
    return text or _FALLBACK


def _format(items: list[dict], urgent: bool) -> str:
    """Envelope + every queued message, in the order the user sent them."""
    body = "\n\n".join(str(i.get("text", "")).strip() for i in items if str(i.get("text", "")).strip())
    return f"{_envelope(urgent)}\n\n{body}"


def _persist(items: list[dict]) -> None:
    """Write the delivered message(s) into the conversation, in the position the
    model read them. The POST that accepted them deliberately does not: one that
    never reaches the agent goes back to the panel to be sent normally, and would
    otherwise be stored twice."""
    tid = interject_bus.thread_id()
    if not tid:
        return
    try:
        from src.utils import conversation_store as cs
        for item in items:
            text = str(item.get("text", "")).strip()
            if text:
                cs.add_message(tid, "user", text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not persist interjection into thread %s: %s", tid, exc)


class InterjectHookProvider(HookProvider):
    """Hands the orchestrator whatever the user said while it was working."""

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:  # noqa: ARG002
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    # ── urgent: cancel the pending call so the message is read first ─────────
    def _on_before(self, event: BeforeToolCallEvent, **kwargs: Any) -> None:  # noqa: ARG002
        try:
            if not interject_bus.has_urgent():
                return
            items = interject_bus.drain()
            if not items:
                return
            name = (getattr(event, "tool_use", None) or {}).get("name", "tool")
            event.cancel_tool = _format(items, urgent=True)
            _persist(items)
            _push_progress(f"🗣 Urgent interjection delivered — cancelled `{name}` so the agent reads it first.")
            logger.info("urgent interjection delivered (%d message(s)); cancelled tool %s", len(items), name)
        except Exception as exc:  # noqa: BLE001
            # A hook that raises would take the turn down with it; a message that
            # misses this boundary is picked up at the next one, or handed back at
            # close_run. Never worth failing the run for.
            logger.warning("interjection (before-tool) failed: %s", exc, exc_info=True)

    # ── normal: ride along with the result of the call that just finished ────
    def _on_after(self, event: AfterToolCallEvent, **kwargs: Any) -> None:  # noqa: ARG002
        try:
            if not interject_bus.pending_count():
                return
            items = interject_bus.drain()
            if not items:
                return
            result = getattr(event, "result", None)
            if not isinstance(result, dict):
                return
            content = list(result.get("content") or [])
            content.append({"text": _format(items, urgent=False)})
            # `result` is one of the two writable fields on this event; replace it
            # wholesale so the ToolResult stays a plain dict of the expected shape.
            event.result = {**result, "content": content}
            _persist(items)
            _push_progress(f"🗣 Interjection delivered to the agent ({len(items)} message(s)).")
            logger.info("interjection delivered (%d message(s)) with the result of %s",
                        len(items), (getattr(event, "tool_use", None) or {}).get("name", "tool"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("interjection (after-tool) failed: %s", exc, exc_info=True)
