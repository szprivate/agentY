"""Stop between two stages and let the user choose what goes on to the next one.

A chain that generates references and then feeds them into a video runs the whole
way through, every time. That is fine when the references are good and expensive
when they are not — the video is the costly half, and by the time you see the
references it has already been paid for.

A ``review`` hook is a break in the chain. The stage before it runs, what it
produced is collected, and the turn ENDS there with the question put to the user.
They look at the images on their canvas, decide which ones deserve the next
stage, and say continue — or stop, and nothing else runs.

**The choice lives on the canvas, not in here.** The halt fills an
``agentY image collector`` node with everything the stage produced and wires it
into the review hook's anchor; whatever is in that node when the user says
continue is what proceeds. That is deliberate and it is the whole design:

* it is *editable* — remove a row, add one of your own, reorder them — which is
  what a review is for, and which a list of checkboxes would not give you;
* it is *persistent* — it survives a click elsewhere, a reload, a week;
* it is *already the input to the next stage* — collector → expander → numbered
  image slots is exactly how a multi-reference video stage is wired, so the thing
  you edit IS the thing that runs, with no selection-to-slot translation to get
  wrong;
* and it means **resume reads the current canvas** rather than a remembered plan.
  The user is expected to edit the graph while it is paused — that is the point —
  and any state we cached at halt time is a chance to act on a graph that no
  longer exists.

So what this module holds is only the *flag*: that a halt is live, which hook it
belongs to, and which collector to read. The answer itself is never in here.

Shaped deliberately like :mod:`src.utils.plan_gate`, which solves the neighbouring
problem (wait for a yes BEFORE anything runs, where this waits between stages):
same refusal-as-a-pause contract for the execution tools, same "their next message
re-opens it" resumption, same insistence that a hold is never reported as a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewHalt:
    """A chain stopped at a review hook, waiting for the user to choose."""

    hook_node_id: str
    # A stable identity for the collector node, stamped onto it as a property so a
    # second halt at the same hook REUSES one node instead of leaving a pile of
    # stale ballots. Deliberately not a node id: the node is created in the
    # browser and litegraph assigns its real id there, where the server cannot see
    # it. The real id is resolved at resume, off the hook's anchors — see
    # :func:`canvas_hooks.review_collector`.
    collector_key: str = ""
    # What the stage produced, as it stood AT THE HALT. Kept for the message the
    # user reads and for the log — never as the answer. The answer is whatever is
    # in the collector when they reply, which is read off the canvas at resume.
    produced: tuple = ()
    question: str = ""
    # The hooks that have not run yet, so the user can be told what continuing buys.
    remaining: tuple = ()

    def count(self) -> int:
        return len(self.produced)

    def describe(self) -> str:
        """One line for the panel: what is waiting and where."""
        n = self.count()
        return (f"hook {self.hook_node_id} — {n} output{'' if n == 1 else 's'} "
                f"waiting for continue or stop")


# The user's reply. Both lists are matched against the WHOLE message: "continue"
# on its own is an answer to the question just put to them, while "continue, but
# make the third one warmer" is a fresh instruction that happens to open with one
# — and that instruction has to reach the agent intact, not be swallowed as a
# bare yes. Anything that is not clearly one or the other is neither, and the
# halt simply stays up.
_CONTINUE = re.compile(
    r"^\s*(?:continue|proceed|carry\s*on|go(?:\s*(?:on|ahead|for\s*it))?|next|"
    r"resume|keep\s*going|do\s*it|run\s*it|send\s*it|ship\s*it|yes|yep|yeah|yup|"
    r"ok(?:ay)?|sure|approved?|confirmed?|lgtm|these|use\s+these|"
    r"continue\s+with\s+these)\b[\s.!]*$", re.I)

_STOP = re.compile(
    r"^\s*(?:stop|halt|abort|cancel|no|nope|discard|drop\s*it|forget\s*it|"
    r"never\s*mind|nevermind|don'?t|do\s*not)\b[\s.!]*$", re.I)

# A longer message still answers if it opens with one of the two and then only
# qualifies it — "continue with these three", "stop, they're all wrong".
_CONTINUE_OPENER = re.compile(
    r"^\s*(?:continue|proceed|carry\s*on|go\s*ahead|resume|keep\s*going)\b", re.I)
_STOP_OPENER = re.compile(r"^\s*(?:stop|abort|cancel|discard|forget\s*it)\b", re.I)
_OPENER_MAX = 120


def read_reply(text: str) -> str:
    """``"continue"`` | ``"stop"`` | ``""`` for the user's answer to a halt.

    Empty means they said something else entirely, which is not an answer: the
    halt stays up and their message is handled as the request it is. Guessing
    here is expensive in the wrong direction — reading "make the third one
    warmer" as a continue spends the video budget nobody approved.
    """
    body = " ".join(str(text or "").split())
    if not body:
        return ""
    if _STOP.match(body):
        return "stop"
    if _CONTINUE.match(body):
        return "continue"
    if len(body) <= _OPENER_MAX:
        if _STOP_OPENER.match(body):
            return "stop"
        if _CONTINUE_OPENER.match(body):
            return "continue"
    return ""


def halt_state(halt: ReviewHalt, collector_node_id: str = "") -> str:
    """The ``[REVIEW HALT]`` block for a turn that begins with one live.

    *collector_node_id* is the ballot's real id, resolved this turn off the hook's
    anchors — the agent needs it to write to the node, and it is not something the
    halt itself can know (see :class:`ReviewHalt`).

    The how-to lives in the ``orchestrator/review_halt`` prompt partial; this is
    only the part that changes from turn to turn.
    """
    where = (f"collector node {collector_node_id}" if collector_node_id
             else "a collector that is no longer wired to the hook")
    head = (f"[REVIEW HALT] The chain is STOPPED at review hook {halt.hook_node_id}. "
            f"{halt.count()} output(s) from the stage before it are waiting in "
            f"{where} on the user's canvas.\n")
    if halt.question:
        head += f"  You asked them: \"{halt.question}\"\n"
    if halt.remaining:
        head += (f"  Not yet run: hook(s) {', '.join(str(h) for h in halt.remaining)}.\n")
    return head + (
        "  The collector is theirs to edit while this is up — rows removed, files "
        "swapped in, order changed. Do NOT re-read what it held when it stopped: "
        "read the node as it stands NOW, and continue with exactly that.\n")


def execution_refusal(halt: ReviewHalt) -> dict:
    """The tool result that stands in for ADVANCING the chain mid-halt.

    Returned by the tools that would queue the stages after the review hook, so
    the agent learns this while it still holds the turn and can put the question
    to the user instead of discovering it afterwards, when only they could.

    It is deliberately not a refusal to *work*: revising what is in the collector
    is the review, and it says so, because an agent told only "no" ends the turn
    and leaves the user with two moves — accept what they have, or throw the run
    away — when what they asked for was a third.
    """
    return {
        "error": (f"not yet — the chain is stopped at review hook {halt.hook_node_id} "
                  f"so the user can choose which of {halt.count()} output(s) go on to "
                  f"the next stage, and they have not answered. Nothing was queued or run."),
        "waiting_on": "the user's continue or stop",
        "what_to_do": (
            "If they asked for a CHANGE to what is in the collector — a different "
            "image, a shorter line, a re-cut clip — do it now: run it inline "
            "(run_workflow_now, apply_canvas_hooks with run_now=True, or "
            "iterate_step) and put the new result into the collector. That is "
            "allowed and is what the stop is for. Only QUEUING the stages after "
            "the hook waits."
        ),
        "then": (
            "End the turn: say what changed, that it is waiting in the collector "
            "on their canvas, and ask whether to continue or stop."
        ),
        "after": (
            "Their next message re-opens this. 'continue' runs the remaining stages "
            "with whatever the collector holds AT THAT POINT — read it fresh, they "
            "have probably edited it. 'stop' ends the run and nothing else is queued."
        ),
        "do_not": "Do not report this as a failure — nothing failed. It is a pause.",
    }


def binding_table(files, roles=None) -> str:
    """The collector's contents rendered as the numbered slots they BECOME.

    Printed this way rather than as a plain list because the renumbering is the
    thing that goes wrong. The multi-reference models are driven by ``@imageN``
    markers in the prompt, and the collector is a LIST: drop the second row and
    everything after it moves up a slot, so a table written before the edit now
    names the wrong picture — the video comes back with the ape doing the
    mentor's beat, and nothing anywhere reports an error.

    Showing the bindings as they will actually be, with each file's own role
    beside it, makes that mismatch visible at the point it has to be fixed
    instead of at the point it renders.
    """
    roles = roles or {}
    lines = []
    for i, path in enumerate(files or [], 1):
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        role = str(roles.get(path) or "").strip()
        lines.append(f"    @image{i} / image_{i} = {name}"
                     + (f" — {role}" if role else ""))
    return "\n".join(lines)


def renumber_note() -> str:
    """The rule that has to survive an edited collector, stated where it applies."""
    return (
        "  These bindings are the ones that will exist, in this order. If your prompt "
        "carries a reference assignment table (@image1 = …, @image2 = …), REWRITE it "
        "to match them: @image2 means the second line as it stands NOW, not the second "
        "thing the earlier stage generated. Dropping one row moves every row after it "
        "up a slot, and a table written before that edit names the wrong picture — "
        "which renders as the wrong character and reports no error.\n")


def resumed_note(kept: int, dropped: int) -> str:
    """What to tell the agent when a halt has just been released with a continue.

    Ends on the precedence rule, because the common continue is not a bare one: a
    user who wants two of five will often say "continue, but drop the second"
    rather than edit the node, and reading that as a plain yes runs the one they
    just told you to leave out.
    """
    if dropped > 0:
        head = (f"The user continued with {kept} of the {kept + dropped} output(s) — "
                f"they removed {dropped}, so do not regenerate those, and RENUMBER any "
                f"@imageN reference table against what is left.")
    else:
        head = f"The user continued with all {kept} output(s)."
    return head + (" This list is the default answer; if their message ALSO says "
                   "something about the selection, that wins — apply it, then say "
                   "which files you ran with.")
