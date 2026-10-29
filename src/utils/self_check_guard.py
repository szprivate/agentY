"""No looking at your own renders to check them while output checks are off.

Seen on 2026-10-29 (qwen3.8-flash as orchestrator): with Settings ▸ Output
checks ▸ qa ▸ enabled switched OFF, the orchestrator still ended turn after turn
with ``analyze_image`` on everything it had just rendered — "is it 16:9? one
bear? rate 1-10" — a vision call the user had turned off, and had to say "dont
analyze the images right now pls" in the middle of the turn to stop it. The
host's own QA honoured the switch (``src/utils/qa.py``); this one was the model
doing QA on its own initiative, nudged by tool text like "then QA the outputs
like any render".

Two parts, both only while ``qa.enabled`` is off:

* the ``output_checks_off`` turn guide (a prompt partial) says so in the prompt;
* :class:`SelfCheckGuardHookProvider` refuses an ``analyze_image`` /
  ``analyze_video`` call whose files are ALL outputs this turn produced, unless
  the user asked to look — in their message or in something they sent mid-turn.

A call that also carries a file that is not a fresh output (a reference, an
attachment, an earlier result) is a comparison or a look at an input, not a
self-check, and goes through: that is how a skill compares a render with the
photo it should match. The refusal is an ordinary tool result (Strands'
``cancel_tool``), so the agent reads it and reports instead.

The pipeline puts what the guard needs on the turn's scope (:func:`begin_turn`);
outside a turn there is nothing to compare with and every call goes through.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("agentY.self_check_guard")

GUARDED = frozenset({"analyze_image", "analyze_video"})
SCOPE_SLOT = "self_check"

# The user asking for a look. Broad on purpose: a false match only lets through a
# call that would have run anyway before this guard existed.
_ASKED = re.compile(
    r"\b(?:look(?:ing)?\s+(?:at|over|through)|(?:have|take)\s+a\s+look|analy[sz]\w*|"
    r"describe|check(?:s|ed|ing)?|judge\w*|compare\w*|comparison|review\w*|inspect\w*|"
    r"verify|rate\s+(?:it|them|each|the\s+\w+)|rating|evaluat\w*|assess\w*|critique|qa|"
    r"which\s+(?:one|is|of)|best\s+one|"
    r"how\s+(?:does|do|did)\s+(?:it|they)\s+look|what\s+do\s+you\s+(?:think|see)|"
    r"is\s+it\s+(?:good|right|ok|okay)|does\s+it\s+match|match(?:es)?\s+the\s+ref\w*|"
    r"schau\w*|anschau\w*|pr[üu]f\w*|bewert\w*|vergleich\w*)\b",
    re.IGNORECASE)


def checks_on() -> bool:
    """Whether output checks are switched on (``qa.enabled``, ``AGENTY_QA``)."""
    try:
        from src.utils.qa import qa_settings
        return bool(qa_settings()["enabled"])
    except Exception:  # noqa: BLE001 - an unreadable setting keeps the old behaviour
        return True


def asked_to_look(text: str) -> bool:
    return bool(_ASKED.search(str(text or "")))


def _norm(path: str) -> str:
    p = str(path or "").strip().strip('"').strip("'")
    if not p:
        return ""
    return os.path.normcase(os.path.normpath(p))


def call_files(args) -> list[str]:
    """The local files an analyze call names (``image_url`` is never an output)."""
    if not isinstance(args, dict):
        return []
    files = [args.get("file_path")]
    many = args.get("file_paths")
    if isinstance(many, (list, tuple)):
        files.extend(many)
    elif isinstance(many, str):
        files.append(many)
    return [str(f) for f in files if str(f or "").strip()]


def refusal(name: str, args, outputs, user_texts=(), *, enabled: bool | None = None) -> str:
    """Why this call is not run, or ``""``. Pure apart from reading the switch."""
    if name not in GUARDED:
        return ""
    if enabled is None:
        enabled = checks_on()
    if enabled:
        return ""
    if isinstance(args, dict) and str(args.get("image_url") or "").strip():
        return ""
    files = call_files(args)
    fresh = {_norm(p) for p in (outputs or []) if _norm(p)}
    if not files or not fresh or not all(_norm(f) in fresh for f in files):
        return ""
    if any(asked_to_look(t) for t in (user_texts or ())):
        return ""
    return (f"Not run. Output checks are switched off in the user's Settings "
            f"(Output checks ▸ qa ▸ enabled), and {name} here would only look at what "
            "this turn just generated. Do not check, rate or compare your own results "
            "unless the user asks you to. Report what was made and where it is, and end "
            "the turn.")


def begin_turn(user_text: str, outputs) -> None:
    """Put this turn's message and its live output list on the turn's scope.

    *outputs* is a callable returning the paths produced so far, read at each
    call — the list grows while the turn runs.
    """
    try:
        from agenty_core.utils import turn_scope
        turn_scope.current().set(SCOPE_SLOT, {"user_text": str(user_text or ""),
                                              "outputs": outputs})
    except Exception as exc:  # noqa: BLE001 - a guard must never cost a turn
        logger.warning("self-check guard: could not record the turn (%s)", exc)


def interjections(messages) -> list[str]:
    """What the user sent into this turn while it ran (``[USER INTERJECTION`` blocks).

    Read from the end back to the message that opened the turn — the last user
    message that is not only tool results.
    """
    found: list[str] = []
    for msg in reversed(list(messages or [])):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content") or []
        opened = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if "toolResult" in block:
                for inner in (block["toolResult"].get("content") or []):
                    t = inner.get("text") if isinstance(inner, dict) else None
                    if isinstance(t, str) and "USER INTERJECTION" in t:
                        found.append(t)
            elif isinstance(block.get("text"), str):
                if "USER INTERJECTION" in block["text"]:
                    found.append(block["text"])
                else:
                    opened = True
        if opened:
            break
    return found


class SelfCheckGuardHookProvider:
    """The orchestrator's: refuses a self-check while output checks are off."""

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ANN001, ARG002
        from strands.hooks.events import BeforeToolCallEvent
        registry.add_callback(BeforeToolCallEvent, self._on_before)

    def _on_before(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            tu = getattr(event, "tool_use", None) or {}
            name = str(tu.get("name", ""))
            if name not in GUARDED:
                return
            from agenty_core.utils import turn_scope
            turn = turn_scope.current().get(SCOPE_SLOT)
            if not isinstance(turn, dict):
                return
            outputs = turn.get("outputs")
            outputs = list(outputs() if callable(outputs) else (outputs or []))
            texts = [turn.get("user_text", "")]
            agent = getattr(event, "agent", None)
            texts += interjections(getattr(agent, "messages", None))
            why = refusal(name, tu.get("input", {}) or {}, outputs, texts)
            if why:
                logger.info("self-check guard: refused %s on %d fresh output(s)",
                            name, len(call_files(tu.get("input") or {})))
                event.cancel_tool = why
        except Exception as exc:  # noqa: BLE001 - a broken guard is an open guard
            logger.warning("self-check guard failed (allowing): %s", exc)
