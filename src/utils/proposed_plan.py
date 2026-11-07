"""The plan the agent last put to the user, for the panel's plan overview.

The panel's Plan button opened the run ledger only (:mod:`src.utils.run_ledger`),
which exists once a hook pipeline is running. A plan the agent *wrote* — the user
asked "make me a plan", a ``[PLAN APPROVAL]`` turn ended on one, the agent
announced its steps before starting, ``run_planner`` broke a request down — was
only ever in the chat, and the overview said there was nothing to show.

So the host keeps the latest one with the conversation, at the end of the turn
that made it:

* from the turn's reply, when it holds a numbered list that is a plan — the list
  comes right after words about a plan or steps, or the user asked for a plan,
  or the turn was holding for a plan approval, or ``run_planner`` ran;
* otherwise from ``run_planner``'s own steps, when it ran this turn.

A newer plan replaces the older one. Stored as::

    {"title": str, "steps": [{"text": str, "detail": str}], "source": "reply"|"planner",
     "updated": <time>}
"""

from __future__ import annotations

import json
import re
import time

_KEY = "proposed_plan"
_MAX_STEPS = 30
_MAX_TEXT = 240
_MAX_DETAIL = 600

# "1. Foo", "2) Foo", "**3.** Foo", "### 4. Foo", "Step 5: Foo", "**Step 6 —** Foo"
_ITEM = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:[*_]{1,2})?\s*(?:(?:step|schritt)\s+)?(\d{1,2})"
    r"\s*(?:[.):]|\s[—–-])(?:[*_]{1,2})?\s+(.+?)\s*$", re.I)
# Words that make the list before them a plan.
_PLAN_WORD = re.compile(r"\b(?:plan|plans|planned|steps?|schritte?|roadmap|approach|"
                        r"here'?s\s+what\s+i(?:'ll|\s+will)|i(?:'ll|\s+will)\s+(?:now\s+)?(?:do|start))\b",
                        re.I)
# The user asking for one.
_ASKS_PLAN = re.compile(r"\b(?:plan|planen|plane)\b", re.I)


def _clean(text: str) -> str:
    """A step's words without markdown emphasis or code ticks."""
    t = re.sub(r"(\*\*|__|`)", "", str(text or ""))
    t = re.sub(r"^\s*[*_]\s*|\s*[*_]\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip(" :-—–")
    return t[:_MAX_TEXT]


def _lists(text: str) -> list[dict]:
    """Every numbered list in *text*: ``{"lead": [lines], "items": [{text, detail}]}``.

    A list runs 1, 2, 3 … in order; lines between two items (bullets, a
    paragraph) are the earlier item's detail. A "1." again starts a new list.
    After the last item, only what hangs off it (indented, a bullet, or before
    the first blank line) is its detail: a closing "Shall I go ahead?" is not.
    ``lead`` is the two non-empty lines before its first item.
    """
    lines = str(text or "").splitlines()
    out: list[dict] = []
    cur = None
    buf: list[str] = []
    in_code = False

    def _flush(final: bool) -> None:
        if cur is None or not buf:
            buf.clear()
            return
        keep = buf
        if final:
            keep = []
            blank = False
            for ln in buf:
                if not ln.strip():
                    blank = True
                    continue
                if blank and not re.match(r"^\s+\S|^\s*[-*•]\s", ln):
                    break
                keep.append(ln)
        cur["items"][-1]["detail"].extend(
            _clean(re.sub(r"^\s*[-*•]\s+", "", ln)) for ln in keep if ln.strip())
        buf.clear()

    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _ITEM.match(line)
        n = int(m.group(1)) if m else 0
        if m and n == 1:
            _flush(final=True)
            before = [ln.strip() for ln in lines[:i] if ln.strip()][-2:]
            cur = {"lead": before, "items": [], "next": 2}
            cur["items"].append({"text": _clean(m.group(2)), "detail": []})
            out.append(cur)
            continue
        if m and cur is not None and n == cur["next"]:
            _flush(final=False)
            cur["items"].append({"text": _clean(m.group(2)), "detail": []})
            cur["next"] += 1
            continue
        if cur is not None:
            buf.append(line)
    _flush(final=True)
    for lst in out:
        lst.pop("next", None)
        for it in lst["items"]:
            it["detail"] = "\n".join(d for d in it["detail"] if d)[:_MAX_DETAIL]
    return [lst for lst in out if len(lst["items"]) >= 2]


def from_reply(reply: str, wanted: bool = False) -> dict | None:
    """The plan in a reply, or None.

    A list right after words about a plan is one whatever the turn was. When the
    turn was about a plan anyway (*wanted*: the user asked for one, a plan
    approval was pending, the planner ran), the longest list is.
    """
    lists = _lists(reply)
    if not lists:
        return None
    chosen = next((lst for lst in lists if _PLAN_WORD.search(" ".join(lst["lead"]))), None)
    if chosen is None and wanted:
        chosen = max(lists, key=lambda lst: len(lst["items"]))
    if chosen is None:
        return None
    # The line right above the list names it when it is a heading or ends in ":".
    last = chosen["lead"][-1] if chosen["lead"] else ""
    title = ""
    if last.startswith("#") or last.rstrip("*_ ").endswith(":"):
        title = _clean(re.sub(r"^#+\s*", "", last)).rstrip(":").strip()
        title = re.split(r"(?<=[.!?])\s+", title)[-1]
        # "Here's the plan" says nothing the overview does not already say.
        if len(title.split()) <= 4 and _PLAN_WORD.search(title):
            title = ""
    return {"title": title[:120], "steps": chosen["items"][:_MAX_STEPS], "source": "reply"}


def from_planner(raw: str) -> dict | None:
    """``run_planner``'s ``{"steps": [{"description", "request"}]}`` as a plan."""
    raw = str(raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    steps = []
    for s in (data.get("steps") if isinstance(data, dict) else None) or []:
        if not isinstance(s, dict):
            continue
        text = _clean(s.get("description") or s.get("request") or "")
        if text:
            req = _clean(s.get("request") or "")
            steps.append({"text": text, "detail": req if req != text else ""})
    if len(steps) < 1:
        return None
    return {"title": "", "steps": steps[:_MAX_STEPS], "source": "planner"}


def asks_for_plan(user_text: str) -> bool:
    return bool(_ASKS_PLAN.search(str(user_text or "")))


def record_turn(thread_id: str, user_text: str, reply: str, planner_raw: str = "",
                awaiting_approval: bool = False) -> dict | None:
    """Keep the plan this turn made, if it made one. Returns what was stored."""
    if not thread_id:
        return None
    wanted = bool(planner_raw) or awaiting_approval or asks_for_plan(user_text)
    plan = from_reply(reply, wanted=wanted) or from_planner(planner_raw)
    if not plan:
        return None
    plan["updated"] = time.time()
    try:
        from src.utils import conversation_store as cs
        cs.kv_set(thread_id, _KEY, plan)
    except Exception:  # noqa: BLE001
        return None
    return plan


def get(thread_id: str) -> dict | None:
    if not thread_id:
        return None
    try:
        from src.utils import conversation_store as cs
        got = cs.kv_get(thread_id, _KEY, None)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(got, dict) or not got.get("steps"):
        return None
    return got


def clear(thread_id: str) -> None:
    if not thread_id:
        return
    try:
        from src.utils import conversation_store as cs
        cs.kv_set(thread_id, _KEY, None)
    except Exception:  # noqa: BLE001
        pass
