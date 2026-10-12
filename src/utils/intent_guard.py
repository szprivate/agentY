"""A turn that ends on "let me do X:" - and then does not do it.

Seen with a local orchestrator (qwen3.6:27b on Ollama, benchmark 2026-10-12): in
ten of twenty-three failed attempts the model wrote what it was about to do -
"Let me look more broadly:", "Let me do this step by step:" - and ended the turn
there, with no tool call after it. The user is left with a promise and nothing
built. A hosted model does it too, only rarely.

The harness can see this for what it is: the turn's last words announce an
action, and no action follows. :func:`unfinished` reads those last words;
the pipeline sends the turn back once with a short note (the orchestrator
partial ``unfinished_intent``) instead of handing a half-step to the user.

Deliberately narrow. It looks only at how the answer ENDS, and only for the two
shapes that were seen: a closing colon, and a last sentence that begins an action
with "let me". A question to the user, "let me know", or a plain statement of
what was done is left alone - sending a finished answer back costs a model call
and makes the agent repeat itself.
"""

from __future__ import annotations

import re

MAX_REDRIVES = 2

# What the runtime itself appends to an answer; not the agent's words.
_RUNTIME = re.compile(r"\n\s*(?:\U0001f9ea|\u2705|\u26a0\ufe0f|\u274c|\u270b)[^\n]*$")
_LET_ME = re.compile(r"^\W*(?:ok(?:ay)?[,.! ]+|now[,.! ]+|first[,.! ]+|next[,.! ]+|so[,.! ]+)*let me\b", re.I)
_LET_ME_KNOW = re.compile(r"\blet me know\b|\blet me say\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# "Now let me make all the edits: 1. Set batch_size ... 2. Add ..." - an announced
# plan, written out and then not carried out. The announcement has to be one: a
# report ("Two things changed: - ...") also ends on a list.
_PLAN = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*((?:[^.!?\n]*?\b(?:let me|i['\u2019]ll|i will|i need to|i['\u2019]m going to|i am going to)\b"
    r"[^.!?\n]*):)\s*(?:\n|\s)*(?:1[.)]|[-*\u2022])\s+\S[\s\S]*$", re.I)


def _clean(text: str) -> str:
    """*text* without the lines the runtime appended to it."""
    t = str(text or "").strip()
    while True:
        cut = _RUNTIME.sub("", t).strip()
        if cut == t:
            return t
        t = cut


def last_words(text: str) -> str:
    """The answer's last sentence (or line), without what the runtime added."""
    t = _clean(text)
    parts = [p.strip() for p in _SENTENCE.split(t) if p.strip()]
    return parts[-1] if parts else ""


def unfinished(text: str) -> str:
    """The sentence that announces an action nothing followed, or ``""``.

    An empty answer counts: a turn that says nothing and calls nothing is the
    same stall without the words.
    """
    last = last_words(text)
    if not last:
        return "(nothing was said)"
    if last.endswith("?") or _LET_ME_KNOW.search(last):
        return ""
    if last.endswith(":"):
        return last
    if _LET_ME.search(last):
        return last
    plan = _PLAN.search(_clean(text))
    if plan and not _LET_ME_KNOW.search(plan.group(1)):
        return plan.group(1).strip()
    return ""


def last_answer(messages) -> str | None:
    """The text of the conversation's last message when it is the agent's own
    closing answer; ``None`` when the turn did not end on one (it ended on a
    tool call or a tool result, so there is nothing to judge)."""
    for msg in reversed(list(messages or ())):
        if not isinstance(msg, dict):
            return None
        if msg.get("role") != "assistant":
            return None
        content = msg.get("content") or []
        if any(isinstance(b, dict) and "toolUse" in b for b in content):
            return None
        return "".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
    return None
