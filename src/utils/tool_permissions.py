"""Ask a person before the agent runs something it cannot be talked out of.

:mod:`agenty_core.sandbox` narrows ``run_script`` to a named list of programs
inside a named list of folders. That is worth having and it is not containment:
``python`` is on the list because skills need it, and a Python process can do
anything Python can do. The honest boundary is not a filter — it is a person
looking at the command.

So the tools that can act irreversibly stop and ask. Modelled on Claude Code,
because that shape has been tested on a lot of people: show the exact thing about
to happen, offer *allow once* and *allow for this session*, and default to no.

**Why a long-poll and not the turn's own stream.** The tool call happens inside
the agent's loop, several frames below the generator that feeds the chat. A hook
that blocked there while publishing through that generator would be holding the
loop that was supposed to deliver its question — the deadlock
:mod:`src.utils.canvas_probe` hit first and documents at length. This channel is
independent of the turn for the same reason, and this module is deliberately
shaped like that one.

**Why the default is deny.** A prompt nobody answers means nobody is watching: the
panel is closed, the turn came from Slack, or the machine was left alone. Every
one of those is a worse moment than usual to run an unreviewed command, so the
timeout resolves the way the absent person would have wanted. It is configurable
for the case where somebody genuinely wants an unattended agent, and saying it out
loud in the settings is better than a default that quietly assumes it.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass

# Tools that stop and ask, unless the settings say otherwise.
#
# Not a list of "dangerous" tools — a list of tools whose effects leave this
# process. Reading a file, searching the web and editing a graph are all
# recoverable; running a program, evaluating Python and installing a package from
# the internet are not, and the last of those runs code from a stranger.
DEFAULT_ASK_TOOLS = ("run_script", "iterate", "install_custom_node")

# How long a question stays on screen before it answers itself.
DEFAULT_TIMEOUT = 120.0

# A question handed to the panel is held back this long before another poll can
# take it, so a reload does not hand the same one to two places — and so a page
# that vanished mid-question releases it rather than stranding the waiter.
_RESERVE_SECONDS = 6.0

_LOCK = threading.Lock()
_PENDING: dict[str, dict] = {}
_SERVED: dict[str, float] = {}
# Set when a question is queued, so a waiting poll wakes at once instead of on its
# next tick. Without it the panel has to choose between a fast poll (two requests
# a second through the access log, every one of them a preflight as well, because
# the token header makes even a GET non-simple) and a prompt that appears a beat
# after the agent stopped.
_ARRIVED = threading.Event()
# Tools the user said yes to for the rest of this host's life.
_SESSION_GRANTS: set[str] = set()
# When the panel last asked for a question. Distinguishes "nobody answered" from
# "nobody was there", which are the same event and want different words.
_last_poll: float = 0.0


@dataclass(frozen=True)
class Decision:
    """What came back. ``allowed`` is the only field the caller must honour."""

    allowed: bool
    reason: str = ""
    remembered: bool = False


def _now() -> float:
    return time.time()


def describe(tool_name: str, tool_input: dict) -> str:
    """One line saying what is about to happen, in the words of the thing itself.

    The command, not a paraphrase of the command. A prompt that summarised would
    be asking the user to trust the summary, which is the opposite of the point.
    """
    data = tool_input if isinstance(tool_input, dict) else {}
    if tool_name == "run_script":
        return str(data.get("command", "")).strip()
    if tool_name == "iterate":
        times = data.get("iter", 1)
        return f"{str(data.get('python_call', '')).strip()}   ×{times}"
    if tool_name == "install_custom_node":
        src = str(data.get("source", "")).strip()
        pip = "" if data.get("run_pip") is False else "  (and pip install its requirements)"
        return f"git clone {src}{pip}"
    try:
        return json.dumps(data, ensure_ascii=False)[:400]
    except (TypeError, ValueError):
        return str(data)[:400]


def reset_session() -> None:
    """Forget every "allow for this session". Called when the host restarts."""
    with _LOCK:
        _SESSION_GRANTS.clear()


def granted_for_session() -> list[str]:
    with _LOCK:
        return sorted(_SESSION_GRANTS)


def has_listener(within: float = 30.0) -> bool:
    """Has the panel asked for questions recently enough to answer one?"""
    with _LOCK:
        return _last_poll > 0 and (_now() - _last_poll) <= within


def request(tool_name: str, tool_input: dict, *, timeout: float = DEFAULT_TIMEOUT,
            unattended_allows: bool = False) -> Decision:
    """Ask, and block this thread until somebody answers.

    Returns a :class:`Decision`; a refusal is an ordinary outcome, not an error.
    """
    name = str(tool_name or "tool")
    with _LOCK:
        if name in _SESSION_GRANTS:
            return Decision(True, "allowed for this session", remembered=True)

    if not has_listener():
        if unattended_allows:
            return Decision(True, "no panel is open and unattended runs are allowed")
        return Decision(False, (
            "this needs your approval and no agentY panel is open to ask. Open "
            "ComfyUI's agentY tab and try again, or set "
            "security.unattended_tool_policy = \"allow\" if this agent is meant to "
            "run without anybody watching."))

    pid = uuid.uuid4().hex[:12]
    event = threading.Event()
    entry = {
        "event": event, "reply": None,
        "request": {"permission_id": pid, "tool": name,
                    "summary": describe(name, tool_input),
                    "asked_at": _now()},
    }
    with _LOCK:
        _PENDING[pid] = entry
    _ARRIVED.set()
    try:
        if not event.wait(timeout=max(1.0, float(timeout or 0))):
            return Decision(False, (
                f"nobody approved this within {int(timeout)}s, so it was not run. "
                "Say what you were trying to do and I can ask again."))
        reply = entry["reply"] if isinstance(entry["reply"], dict) else {}
        allowed = bool(reply.get("allowed"))
        if allowed and reply.get("remember"):
            with _LOCK:
                _SESSION_GRANTS.add(name)
            return Decision(True, "allowed for the rest of this session", remembered=True)
        if allowed:
            return Decision(True, "allowed once")
        note = str(reply.get("note") or "").strip()
        return Decision(False, note or "you declined this.")
    finally:
        with _LOCK:
            _PENDING.pop(pid, None)
            _SERVED.pop(pid, None)


def _sweep(now: float) -> None:
    for pid, at in list(_SERVED.items()):
        if now - at > _RESERVE_SECONDS or pid not in _PENDING:
            _SERVED.pop(pid, None)


def _take_now() -> dict | None:
    now = _now()
    with _LOCK:
        _sweep(now)
        for pid, entry in _PENDING.items():
            if pid in _SERVED:
                continue
            _SERVED[pid] = now
            return dict(entry["request"])
    return None


def take(wait: float = 0.0) -> dict | None:
    """The next question waiting, or None once *wait* seconds have passed.

    A real long poll: the connection is held open until there is something to say.
    The alternative — the panel asking every second or so — costs two requests a
    second (the token header forces a CORS preflight before each one), fills the
    host's console, and is still slower to show the prompt than this.

    It ticks rather than sleeping the whole time because a question can become
    available without anything being queued: one handed to a page that then
    vanished is held back only until its reservation lapses.
    """
    global _last_poll
    # Marked on the way IN. has_listener() is read while this call is blocked, and
    # a panel that is waiting is the most present a panel ever is — recording it
    # on the way out would call it absent for the whole time it was listening.
    with _LOCK:
        _last_poll = _now()

    deadline = _now() + max(0.0, float(wait or 0))
    while True:
        found = _take_now()
        if found is not None:
            return found
        remaining = deadline - _now()
        if remaining <= 0:
            return None
        _ARRIVED.clear()
        _ARRIVED.wait(timeout=min(1.0, remaining))


def answer(permission_id: str, allowed: bool, *, remember: bool = False,
           note: str = "") -> bool:
    """Deliver the user's answer. False if that question is no longer waiting."""
    with _LOCK:
        entry = _PENDING.get(str(permission_id or ""))
        if entry is None:
            return False
        entry["reply"] = {"allowed": bool(allowed), "remember": bool(remember),
                          "note": str(note or "")}
        entry["event"].set()
        return True


def pending_count() -> int:
    with _LOCK:
        return len(_PENDING)
