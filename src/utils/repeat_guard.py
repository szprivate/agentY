"""Stop an agent that keeps making the call it has already made.

Seen on 2026-10-12 (qwen3.8:27b as orchestrator, benchmark case CX04): looking
for a depth ControlNet, the model called ``hf_file`` 136 times in one turn - two
URLs, 63 and 62 times each, every answer identical to the one before. Nothing
stopped it. The history outgrew the model's window, the server cut the start of
it off (the user's request included) and answered "no user query found", and the
retry, in the same conversation, died at once.

Two rules, both about what actually happened in the turn, not about which model
is running:

**The same call, the same answer.** A call whose tool and arguments are exactly
those of an earlier call in this turn, and whose earlier answers were all the
same, is refused once it has been answered ``same_call`` times (3). Nothing new
can come back. Tools whose answer is SUPPOSED to change while the arguments stay
the same - the queue, a download's progress, a shot's status - are compared like
any other: while the answer keeps changing they are never refused, and they are
given more patience when it does not (``POLLING``), because waiting on a render
is what polling looks like.

**One tool, over and over.** A tool called ``per_tool`` times in one turn (30),
whatever the arguments, gets a single refusal that says so and asks the model to
decide with what it has. It is a speed bump, not a wall: the count has to double
before the next one, and a folder of forty reference images analysed one by one
passes with one extra step. After the third bump the tool is closed for the rest
of the turn.

A refusal is an ordinary tool result (Strands' ``cancel_tool``): the agent reads
it and goes on, which is what lets it recover rather than fail.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger("agentY.repeat_guard")

SAME_CALL = 3
PER_TOOL = 30
MAX_BUMPS = 3
# Asked again with the same arguments on purpose, while something else works.
POLLING = frozenset({
    "queue", "get_history", "get_prompt_status_by_id", "wait_for_download", "shot_status",
    "get_batch_status", "list_batch_jobs", "get_system_stats", "loop_check", "list_outputs",
    "get_canvas_node", "screenshot_canvas", "get_logs",
})
POLLING_PATIENCE = 12


def settings() -> dict:
    out = {"enabled": True, "same_call": SAME_CALL, "per_tool": PER_TOOL}
    try:
        from src.utils.settings import load_settings
        cfg = load_settings()
        out["enabled"] = bool(cfg.get("repeat_guard", True))
        out["same_call"] = max(2, int(cfg.get("repeat_guard_same_call", SAME_CALL)))
        out["per_tool"] = max(5, int(cfg.get("repeat_guard_per_tool", PER_TOOL)))
    except Exception:  # noqa: BLE001
        pass
    return out


def _digest(value) -> str:
    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = str(value)
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _result_text(result) -> str:
    """What a tool answered, without the parts that differ on every call (ids)."""
    if isinstance(result, dict):
        return json.dumps([result.get("status"), result.get("content")], sort_keys=True, default=str)
    return str(result)


class Turn:
    """What one agent has called so far in one turn. Pure: no agent, no settings."""

    def __init__(self, same_call: int = SAME_CALL, per_tool: int = PER_TOOL):
        self.same_call = same_call
        self.per_tool = per_tool
        self._answers: dict[str, list[str]] = {}       # call -> digests of its answers
        self._count: dict[str, int] = {}               # tool -> calls let through
        self._bumps: dict[str, int] = {}               # tool -> refusals for volume
        self._next_bump: dict[str, int] = {}

    @staticmethod
    def key(name: str, args) -> str:
        return f"{name}:{_digest(args)}"

    def refusal(self, name: str, args) -> str:
        """Why this call is not run, or ``""`` to let it through (and count it)."""
        answers = self._answers.get(self.key(name, args), [])
        patience = max(self.same_call, POLLING_PATIENCE) if name in POLLING else self.same_call
        if len(answers) >= patience and len(set(answers[-patience:])) == 1:
            return (f"Not run. You have made this exact call to {name} {len(answers)} times this turn "
                    f"and the answer was the same every time; asking again will not change it. "
                    f"Use the answer you already have. If it does not give you what you need, "
                    f"that is the finding: try a different approach, or tell the user what you "
                    f"could not find or do.")
        count = self._count.get(name, 0)
        threshold = self._next_bump.get(name, self.per_tool)
        bumps = self._bumps.get(name, 0)
        if bumps >= MAX_BUMPS:
            return (f"Not run. {name} is closed for the rest of this turn: you have called it "
                    f"{count} times. Decide with what you have, or tell the user where you are stuck.")
        if count >= threshold:
            self._bumps[name] = bumps + 1
            self._next_bump[name] = threshold * 2
            return (f"Not run. You have called {name} {count} times this turn. If you are searching "
                    f"or checking in a loop, stop here: decide with what you have, or tell the user "
                    f"what you could not find. If every one of those calls was needed (a long list "
                    f"worked through item by item), carry on - this notice comes once at this count.")
        self._count[name] = count + 1
        return ""

    def answered(self, name: str, args, result) -> None:
        self._answers.setdefault(self.key(name, args), []).append(_digest(_result_text(result)))


def collapse_repeats(messages: list) -> int:
    """Drop the middle copies of a tool round repeated three or more times.

    A round is an assistant message of tool calls (and any words around them) plus the user message
    that carries their results. Of each set of identical rounds (same tools, same
    arguments) the first - the real answer - and the last - usually this guard's
    refusal - stay. DashScope rejects a request whose history holds the same call
    over many rounds ("Repetitive tool calls detected", CX10, night_20261029), and
    the history stays poisoned: the next two turns of that conversation died on
    their first model call. Returns how many rounds were dropped.
    """
    rounds: dict[tuple, list[int]] = {}
    for i, msg in enumerate(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content") or []
        calls = [b for b in content if isinstance(b, dict) and "toolUse" in b]
        if not calls or not all(isinstance(b, dict) and ("toolUse" in b or "text" in b
                                                         or "reasoningContent" in b) for b in content):
            continue
        content = calls
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if not isinstance(nxt, dict) or nxt.get("role") != "user":
            continue
        results = nxt.get("content") or []
        if not results or not all(isinstance(b, dict) and "toolResult" in b for b in results):
            continue
        ids = {str(b["toolUse"].get("toolUseId")) for b in content}
        if {str(b["toolResult"].get("toolUseId")) for b in results} != ids:
            continue
        sig = tuple(sorted((str(b["toolUse"].get("name")), _digest(b["toolUse"].get("input")))
                           for b in content))
        rounds.setdefault(sig, []).append(i)
    drop: set[int] = set()
    for idx in rounds.values():
        if len(idx) >= 3:
            for i in idx[1:-1]:
                drop.update((i, i + 1))
    if drop:
        messages[:] = [m for j, m in enumerate(messages) if j not in drop]
    return len(drop) // 2


def shared_turn(scope: str, cfg: dict | None = None) -> Turn:
    """The :class:`Turn` every agent working on *scope* in this turn shares.

    A fresh ``Turn`` per invocation is right for an agent that owns its work and
    wrong for one restarted over the same job. The repair specialist is built
    anew for each heal of a workflow with its messages cleared, so a
    per-invocation budget lets every attempt start from zero — the loop this
    guard exists to stop is spread across attempts thin enough to never trip it.
    Keyed by the thing being worked on and held in the turn's own scope, so one
    turn's budget is not spent by the next.
    """
    cfg = cfg or settings()
    try:
        from agenty_core.utils import turn_scope
        store = turn_scope.current().slot("repeat_guard.shared", dict)
    except Exception:  # noqa: BLE001 - outside a turn: a private budget is all there is
        return Turn(cfg["same_call"], cfg["per_tool"])
    # setdefault, not get-then-set: heals run concurrently, and two Turns for one
    # workflow would split the budget this exists to keep whole.
    return store.setdefault(scope, Turn(cfg["same_call"], cfg["per_tool"]))


class RepeatGuardHookProvider:
    """One :class:`Turn` per agent invocation; refusals go out as cancelled calls.

    With *scope*, the budget is shared with every other provider naming the same
    scope in this turn instead of restarting per invocation — see
    :func:`shared_turn`.
    """

    def __init__(self, scope: str = "") -> None:
        self._scope = str(scope or "")
        self._turn: Turn | None = None
        self._on = True

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks import BeforeInvocationEvent
        from strands.hooks.events import AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent
        registry.add_callback(BeforeInvocationEvent, self._start)
        registry.add_callback(BeforeToolCallEvent, self._before)
        registry.add_callback(AfterToolCallEvent, self._after)
        registry.add_callback(BeforeModelCallEvent, self._before_model)

    def _before_model(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        """Keep a run of refused repeats out of what the model is sent."""
        try:
            if not self._on:
                return
            msgs = getattr(getattr(event, "agent", None), "messages", None)
            if isinstance(msgs, list):
                n = collapse_repeats(msgs)
                if n:
                    logger.info("repeat guard: dropped %d repeated tool round(s) from the history", n)
        except Exception as exc:  # noqa: BLE001
            logger.debug("repeat guard: collapse skipped (%s)", exc)

    def _start(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        cfg = settings()
        self._on = cfg["enabled"]
        self._turn = (shared_turn(self._scope, cfg) if self._scope
                      else Turn(cfg["same_call"], cfg["per_tool"]))

    def _before(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            if not self._on or getattr(event, "cancel_tool", False):
                return
            if self._turn is None:
                self._start(event)
            use = getattr(event, "tool_use", None) or {}
            why = self._turn.refusal(str(use.get("name", "")), use.get("input"))
            if why:
                event.cancel_tool = why
                logger.info("repeat guard: refused %s", use.get("name"))
        except Exception as exc:  # noqa: BLE001 - a broken guard must not break a turn
            logger.debug("repeat guard skipped (%s)", exc)

    def _after(self, event, **kwargs) -> None:  # noqa: ANN001, ARG002
        try:
            if not self._on or self._turn is None:
                return
            # A call that was not run has no answer. Counting the refusal as one
            # broke the run of identical answers, so the next copy of the call ran
            # again: refused, ran, refused, ran ... (CX10, night_20261029: the
            # same two directory scans 20+ times each, until DashScope refused
            # the whole request).
            if getattr(event, "cancel_message", None):
                return
            use = getattr(event, "tool_use", None) or {}
            self._turn.answered(str(use.get("name", "")), use.get("input"), getattr(event, "result", None))
        except Exception as exc:  # noqa: BLE001
            logger.debug("repeat guard skipped (%s)", exc)
