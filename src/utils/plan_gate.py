"""Say the plan before acting on it — and wait only where waiting was asked for.

Two rules that look alike and are not.

The first is unconditional and lives in the prompts: when a turn is going to take
several steps, the agent writes the plan into the chat **before** it starts, so
the user can see where this is going and interject while it runs instead of after.
That is an announcement, and nothing here enforces it.

The second is the one that needs enforcing: actually **waiting** for a yes. That
happens only where someone asked to be asked — in the user's message, in a hook
node's directive, or in the project's own memory. Making it the default would
turn every multi-step request into a round trip nobody asked for, so this module
decides deterministically whether a turn is one of those and the pipeline holds
the execution tools shut until the user has had their say.

Detection is deliberately narrow, because the neighbouring feature already owns
the other reading of "wait": a hook directive saying *"wait for all the references
to be generated"* is a CONDITIONAL hook (:func:`canvas_hooks.is_conditional`) —
what it waits on there is an outcome, not a person. Every pattern below therefore
has the user in it: *my* go-ahead, ask *me* first, don't start until *I* say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ApprovalRequest:
    """Where the request to be asked came from, and the words that asked."""

    source: str   # "the user's message", "hook 30's directive", "the project's memory"
    quote: str    # the sentence that asked, so the agent can honour what was meant


_ME = r"(?:me|us|the\s+user|user)"
_MY = r"(?:my|our|the\s+user'?s?|users?')"
# Only ever read directly after "my" / "the user's", where a bare "go" or "ok"
# can mean nothing except permission.
_OK = (r"(?:ok(?:ay)?|go|go[-\s]?ahead|approval|sign[-\s]?off|confirmation|"
       r"green\s*light|blessing|permission|say[-\s]?so|nod|yes)")
_START = (r"(?:start|begin|run|running|execut\w*|generat\w*|queue|queu\w+|"
          r"proceed|continu\w*|build|do\s+anything)")

_APPROVAL_PATTERNS = (
    # "wait for my go-ahead", "hold until the user's sign-off", "await my ok"
    re.compile(rf"\b(?:wait|hold|await|pause)\b[^.!?]{{0,24}}\b{_MY}\s+{_OK}\b", re.I),
    # "wait until I say so", "hold until I tell you"
    re.compile(rf"\b(?:wait|hold|await|pause)\b[^.!?]{{0,12}}\b(?:until|till)\s+"
               rf"(?:i|we|{_ME})\b", re.I),
    # "ask me first", "check with me before you run anything"
    re.compile(rf"\b(?:ask|check\s+with|clear\s+it\s+with|confirm\s+with|run\s+it\s+by)\s+"
               rf"{_ME}\b[^.!?]{{0,48}}\b(?:first|before|beforehand)\b", re.I),
    re.compile(rf"\b(?:ask|check\s+with|confirm\s+with)\s+{_ME}\s+(?:first|before)\b", re.I),
    # "before you start, ask me" / "before generating anything, get my ok"
    re.compile(rf"\bbefore\s+(?:you\s+)?{_START}\b[^.!?]{{0,64}}\b(?:ask\s+{_ME}|"
               rf"check\s+with\s+{_ME}|confirm\s+with\s+{_ME}|{_MY}\s+{_OK})\b", re.I),
    # "don't run anything until I approve"
    re.compile(rf"\bdo\s*n[o']?t\s+{_START}\b[^.!?]{{0,64}}\b(?:until|unless|before)\b"
               rf"[^.!?]{{0,32}}\b(?:i|we|{_ME}|{_MY})\b", re.I),
    # "my approval is required", "this needs my sign-off first"
    re.compile(rf"\b{_MY}\s+{_OK}\b[^.!?]{{0,32}}\b(?:required|needed|necessary|first)\b", re.I),
    re.compile(rf"\b(?:needs?|requires?|wants?|get)\b[^.!?]{{0,24}}\b{_MY}\s+{_OK}\b", re.I),
    # "let me approve it", "let the user confirm the plan"
    re.compile(rf"\blet\s+{_ME}\s+(?:approve|confirm|review|decide|sign\s+off)\b", re.I),
    # "show me the plan first", "present the plan to me and wait" — the user and
    # the plan both have to be in the sentence, in either order.
    re.compile(rf"\b(?:show|present|give|send)\b(?=[^.!?]{{0,60}}\b{_ME}\b)"
               rf"[^.!?]{{0,32}}\bplan\b[^.!?]{{0,40}}"
               rf"\b(?:first|before|and\s+wait|then\s+wait)\b", re.I),
)

# The user overruling a standing "ask me first" for this turn. Only ever read
# against the user's own message: a hook node cannot waive the user's rule, and
# the user saying "go ahead" IS the approval the gate was waiting for.
_WAIVER_PATTERNS = (
    re.compile(r"\b(?:do\s*n[o']?t|no\s+need\s+to|never)\s+(?:ask|wait)\b", re.I),
    re.compile(r"\bwithout\s+(?:asking|waiting|checking)\b", re.I),
    re.compile(r"\bno\s+(?:approval|confirmation|sign[-\s]?off|plan)\s+"
               r"(?:needed|required|necessary|this\s+time)\b", re.I),
    re.compile(rf"\byou\s+do\s*n[o']?t\s+need\s+{_MY}\b", re.I),
    re.compile(r"\bgo\s+ahead\b", re.I),
    # The idiom, not any verb after "just": "just make it warmer" is a request,
    # and running it unasked is exactly what the gate exists to prevent.
    re.compile(r"\bjust\s+(?:do\s+it|go(?:\s+ahead)?|run\s+it|get\s+(?:on|going)|"
               r"start|crack\s+on)\b", re.I),
)

# A reply that is nothing but agreement. Kept separate from the waivers above
# because it only counts as one when it is the WHOLE message: "yes" on its own is
# an answer to the plan that was just put to them, while "yes, and make it warmer"
# is a fresh instruction that happens to open with a yes.
_AFFIRMATION = re.compile(
    r"^\s*(?:yes|yep|yeah|yup|ok(?:ay)?|sure|fine|perfect|great|nice|approved?|"
    r"agreed|confirmed?|correct|proceed|continue|carry\s+on|do\s+it|run\s+it|"
    r"send\s+it|go(?:\s+for\s+it)?|sounds?\s+good|looks?\s+good|lgtm)\b", re.I)
_AFFIRMATION_MAX = 40

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentence_around(text: str, pos: int) -> str:
    """The one sentence containing *pos*, whitespace-normalised and trimmed.

    Quoting the sentence rather than the match is what lets the agent honour what
    was actually asked ("show me the plan AND the cost") instead of the two words
    a regex happened to land on.
    """
    start = 0
    for m in _SPLIT.finditer(text):
        if m.end() > pos:
            break
        start = m.end()
    end = len(text)
    for m in _SPLIT.finditer(text, pos):
        end = m.start()
        break
    return " ".join(text[start:end].split())[:220]


def find_approval_request(sources) -> ApprovalRequest | None:
    """The first place in *sources* that asked to approve the plan, or None.

    *sources* is an ordered list of ``(label, text)`` — the user's message first,
    then the hook directives, then the project's memory. First hit wins: they all
    mean the same thing, and one quote is easier to act on than three.
    """
    for label, text in sources or ():
        body = str(text or "")
        if not body.strip():
            continue
        for pat in _APPROVAL_PATTERNS:
            m = pat.search(body)
            if m:
                return ApprovalRequest(source=str(label),
                                       quote=_sentence_around(body, m.start()))
    return None


def waived(user_text: str) -> bool:
    """True when the user's own message overrules a standing "ask me first".

    Two ways in. "Just do it", "no need to ask" — a rule they set is theirs to
    suspend, and the alternative is a user who has to argue with their own hook
    node to get one turn done. And a message that is *only* agreement ("yes",
    "go ahead", "looks good") — that is the answer to the plan they were just
    shown, so treating it as another turn to be gated would ask them twice.
    """
    body = str(user_text or "")
    if any(p.search(body) for p in _WAIVER_PATTERNS):
        return True
    one_line = " ".join(body.split())
    return len(one_line) <= _AFFIRMATION_MAX and bool(_AFFIRMATION.match(one_line))


def approval_state(req: ApprovalRequest, answered: bool) -> str:
    """The ``[PLAN APPROVAL]`` block: who asked, in what words, and where we are.

    The how-to lives in the ``orchestrator/plan_approval`` prompt partial; this is
    only the part that changes from turn to turn.
    """
    head = (f"[PLAN APPROVAL] {req.source} asks to approve the plan before anything "
            f"runs:\n  \"{req.quote}\"\n")
    if answered:
        return head + (
            "  They have since replied — read their message. A yes means carry the "
            "plan out now; a change means apply it and carry on. Do not ask them to "
            "approve the same work twice.\n")
    return head + (
        "  They have NOT answered yet: this turn is for presenting the plan, not for "
        "running it. The execution tools will refuse until they have replied.\n")


def plan_note(req: ApprovalRequest | None = None, answered: bool = False) -> str:
    """What to do with a plan that was just handed back (``run_planner``).

    Appended to the planner's own JSON so the instruction arrives with the thing
    it is about, rather than only in a system prompt written several thousand
    tokens earlier.
    """
    say = ("NEXT: say this plan to the user in the chat before you start on it — one "
           "short numbered line per step, in your own words.")
    if req is None or answered:
        return say + (" It is an announcement, not a question: say it, then get on "
                      "with step 1 in the same turn.")
    return say + (f" Then STOP there — {req.source} asked to approve it first "
                  f"(\"{req.quote}\"). End the turn with the plan and the question; "
                  "do not call an execution tool.")


def execution_refusal(req: ApprovalRequest) -> dict:
    """The tool result that stands in for running something before the user agreed.

    Returned by every tool that would run or queue work, so the agent learns this
    while it still holds the turn and can do the right thing (present the plan)
    instead of afterwards, when only the user could.
    """
    return {
        "error": "not yet — this run was asked to be approved first, and the user "
                 "has not answered. Nothing was queued or run.",
        "asked_by": req.source,
        "their_words": req.quote,
        "what_to_do": (
            "End the turn with the PLAN instead: a short numbered list of the steps "
            "you were about to take (what gets generated, how many, from which "
            "inputs), then ask for their go-ahead. Do not call this or any other "
            "execution tool again this turn."
        ),
        "after": (
            "Their next message re-opens this. A yes means run the plan as stated; a "
            "change means apply it and run — you will not be asked to present it again "
            "for the same work."
        ),
        "do_not": "Do not report this as a failure — nothing failed. It is a pause.",
    }
