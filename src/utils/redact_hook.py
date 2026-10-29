"""Mask secrets in every tool result before the model (and the history) sees it.

The backstop behind the file tools' name checks (agenty_core.utils.path_guard):
a turn did read ``.env`` through read_text_file once, and the keys went to the
model provider and into conversations.sqlite. Whatever path a secret takes into
a tool result - run_script output, a log, an error quoting a URL - it leaves
here masked (agenty_core.utils.redact).

Registered on every agent by src.agent's factory, LAST: AfterToolCallEvent
callbacks run in reverse order, so this one runs first and every other hook
(the activity cards, compaction, the history) gets the masked result.
"""

from __future__ import annotations

import logging

from strands.hooks import HookRegistry
from strands.hooks.events import AfterToolCallEvent

logger = logging.getLogger("agentY.redact")


def redact_tool_result(result):
    """A ToolResult with its text and json blocks redacted; other blocks as they were."""
    from agenty_core.utils.redact import redact, redact_obj
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list):
        return result
    changed = False
    new_blocks = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            text = redact(block["text"])
            if text != block["text"]:
                block = dict(block, text=text)
                changed = True
        elif isinstance(block, dict) and "json" in block:
            js = redact_obj(block["json"])
            if js != block["json"]:
                block = dict(block, json=js)
                changed = True
        new_blocks.append(block)
    if not changed:
        return result
    return dict(result, content=new_blocks)


class RedactSecretsHook:
    """AfterToolCallEvent: replace the result with its redacted copy."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:  # noqa: ARG002
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_after(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            res = getattr(event, "result", None)
            red = redact_tool_result(res)
            if red is not res:
                event.result = red
                tu = getattr(event, "tool_use", None) or {}
                logger.warning("masked a secret in the result of %s", tu.get("name", "a tool"))
        except Exception as exc:  # noqa: BLE001 - never lose a result over this
            logger.warning("secret redaction skipped: %s", exc)
