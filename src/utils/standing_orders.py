"""A user's "decide yourself" stays true for the rest of the conversation.

Seen 2026-10-20 (the "Zilius" restaurant session, local models): the user said
"don't stop, decide yourself", and the agent went on ending turns with a menu -
"Route A / B / C, which do you want?" - because the sentence lived in one message
that later turns (and the summary of old turns) no longer carried. So the
instruction is kept with the conversation (``AgentSession.standing_autonomy``),
re-stated to the agent on every turn, and honoured by the plan gate.

It is NOT a licence to skip questions that are the user's: money, deleting their
work, a choice only they can make (their face, their brand, a gated model they
must unlock). The prompt partial ``autonomy`` says so; this module only decides
what counts as the instruction, when it is lifted, and what a menu looks like.
"""

from __future__ import annotations

import re

# Only ever read against the USER's own message (never a hook directive or the
# harness's text): the user's words are what grant it.
_GRANT = (
    # "don't stop" on its own / "don't stop to ask" - not "don't stop the render"
    re.compile(r"\bdo\s*n[o']?t\s+stop\b(?!\s+(?:the|this|that|it|them|my|your|a|an|any)\b)", re.I),
    re.compile(r"\bstop\s+(?:asking|stopping|checking\s+in|interrupting)\b", re.I),
    re.compile(r"\b(?:do\s*n[o']?t|never)\s+(?:ask|check\s+in)\b[^.!?]{0,24}\b(?:me|again|anything|questions?)\b", re.I),
    re.compile(r"\bno\s+more\s+(?:questions|options|choices|menus?)\b", re.I),
    re.compile(r"\bdecide\s+(?:it\s+|this\s+|that\s+|everything\s+)?(?:for\s+)?yourself\b", re.I),
    re.compile(r"\byou\s+decide\b(?!\s+(?:between|whether|if|which|what|on\s+the))", re.I),
    # "it's your call" / "up to you" - not "your call to the API failed"
    re.compile(r"\b(?:it'?s|its|it\s+is|that'?s)\s+(?:all\s+)?your\s+(?:call|choice|decision)\b", re.I),
    re.compile(r"\b(?:it'?s|it\s+is)\s+(?:all\s+)?up\s+to\s+you\b", re.I),
    re.compile(r"\b(?:pick|choose)\b[^.!?]{0,16}\byourself\b", re.I),
    re.compile(r"\b(?:make|take)\s+(?:the|these|all)\s+(?:decisions?|choices?|calls?)\s+(?:yourself|for\s+me)\b", re.I),
    re.compile(r"\bwithout\s+(?:asking|stopping|checking\s+(?:in|with\s+me))\b", re.I),
    re.compile(r"\b(?:keep\s+going|carry\s+on)\b[^.!?]{0,24}\b(?:until|till)\s+(?:it'?s\s+)?(?:done|finished)\b", re.I),
    re.compile(r"\bdo\s*n[o']?t\s+wait\s+for\s+me\b", re.I),
)

# The user taking the decisions back.
_REVOKE = (
    re.compile(r"\b(?:ask|check\s+with|confirm\s+with)\s+me\b[^.!?]{0,24}\b(?:first|before|again|from\s+now)\b", re.I),
    re.compile(r"\b(?:you\s+may|you\s+can|feel\s+free\s+to)\s+ask\s+(?:me\s+)?(?:again|questions)\b", re.I),
    re.compile(r"\blet\s+me\s+decide\b", re.I),
    re.compile(r"\bi(?:'ll|\s+will)\s+decide\b", re.I),
    re.compile(r"\bstop\s+deciding\b", re.I),
)

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _quote(text: str, pos: int) -> str:
    start = 0
    for m in _SPLIT.finditer(text):
        if m.end() > pos:
            break
        start = m.end()
    end = len(text)
    for m in _SPLIT.finditer(text, pos):
        end = m.start()
        break
    return " ".join(text[start:end].split())[:200]


def granted(user_text: str) -> str:
    """The sentence in which the user told the agent to decide, or ``""``."""
    body = str(user_text or "")
    for pat in _GRANT:
        m = pat.search(body)
        if m:
            return _quote(body, m.start())
    return ""


def revoked(user_text: str) -> bool:
    body = str(user_text or "")
    return any(p.search(body) for p in _REVOKE)


def update(current: str, user_text: str) -> str:
    """The standing instruction after this user message: kept, set, or lifted.

    A message that grants wins over one that revokes: the revoke patterns are
    narrow, and a message saying both ("you decide, I'll check later") is a grant.
    """
    new = granted(user_text)
    if new:
        return new
    if revoked(user_text):
        return ""
    return str(current or "")


# -- a turn that ends on a menu ------------------------------------------------

_OPTION_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*|__)?(?:option|route|path|approach|choice|variant)?\s*"
    r"(?:[A-Da-d]|[1-4])(?:\*\*|__)?\s*[).:—–-]\s*\S", re.I)
_OPTION_WORD = re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*|__)?(?:option|route|path|approach|choice)\s+"
                          r"(?:[A-Da-d]|[1-4])\b", re.I)
_WHICH = re.compile(
    r"\b(?:which\s+(?:one|option|route|path|template|approach|of\s+these)|"
    r"what\s+(?:would\s+you\s+(?:like|prefer)|do\s+you\s+(?:prefer|want))|"
    r"(?:would|do)\s+you\s+(?:like|want|prefer)\s+(?:me\s+to|to\s+(?:go|use|take|pick))|"
    r"(?:shall|should)\s+i\s+(?:go\s+with|use|take|pick)|"
    r"tell\s+me\s+which|let\s+me\s+know\s+which|your\s+(?:pick|choice|call))\b", re.I)


def menu(answer: str) -> str:
    """The closing sentence when *answer* ends by handing the user a menu, or ``""``.

    Needs two or more lettered/numbered options AND a question asking which. A
    numbered list of results with no such question is a report, and is left alone.
    """
    text = str(answer or "").strip()
    if not text:
        return ""
    tail = text[-900:]
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    options = sum(1 for ln in lines if _OPTION_LINE.match(ln) or _OPTION_WORD.match(ln))
    close = tail[-400:]
    asks = bool(_WHICH.search(close)) and ("?" in close or "let me know" in close.lower())
    if options >= 2 and asks:
        parts = [p.strip() for p in _SPLIT.split(tail) if p.strip()]
        return (parts[-1] if parts else "")[:200]
    return ""
