"""Writing a prompt, looking at the render, writing it again.

This is the loop people actually run: ask the agent for a prompt, queue it in
ComfyUI yourself, look at what came out, say what to change, go again. Before this
existed it worked — `set_canvas_node_params` has always been able to write a prompt
into a node — but every round started from nothing: which node held the prompt, what
the last prompt said, which render it produced, what had already been tried and
rejected. All of it lived in the conversation, and the conversation is a window that
slides.

So the loop gets state of its own, per conversation and on disk:

* **the target** — the node and input the prompt is written into, so "make it warmer"
  needs no node id and no selection;
* **the versions** — every prompt the agent has written in this loop, numbered, and
  which of them is **active** (on the canvas now: the newest, or the one the user
  picked in the strip — picking never adds a version);
* **the pairing** — which render came out of which version, found from ComfyUI's own
  history: the run is queued by the panel, in the user's browser, on the user's own
  graph, so agentY never sees it go by.

Deliberately *not* here: running anything. A new version is queued by the panel
pressing ComfyUI's own Queue on the canvas it just wrote into (the ``queue`` flag on
the ``prompt_version`` patch) — the same run the user would have started by hand,
which is what they asked for.
"""

from __future__ import annotations

import time

# Nothing here is a limit on the user; both caps exist so a loop that runs all
# afternoon cannot grow the injected block without bound.
_MAX_VERSIONS = 40
_BLOCK_VERSIONS = 8        # how many are shown to the agent, newest last
_TEXT_IN_BLOCK = 400       # characters of an older version's text in the block
_RESOLVE_TRIES = 4         # newest-first file lookups before giving up (see newest_output)


def _store():
    from src.utils import conversation_store as cs
    return cs


def state(thread_id: str) -> dict | None:
    """This thread's loop, or None when it has none. ``{"on": False, …}`` after a stop."""
    if not thread_id:
        return None
    try:
        return _store().get_prompt_loop(thread_id)
    except Exception:  # noqa: BLE001 — a missing store is not a broken turn
        return None


def active(thread_id: str) -> dict | None:
    """The loop only if it is switched ON, so callers cannot forget to check."""
    loop = state(thread_id)
    return loop if isinstance(loop, dict) and loop.get("on") else None


def _save(thread_id: str, loop: dict | None) -> dict | None:
    try:
        _store().set_prompt_loop(thread_id, loop)
    except Exception:  # noqa: BLE001
        pass
    return loop


def start(thread_id: str, node_id: str = "", input_name: str = "") -> dict:
    """Switch the loop on, keeping any versions it already has.

    Re-starting is not a reset: the usual reason to switch it back on is to carry on
    with the same prompt after doing something else, and throwing away ten versions
    to spare one line of bookkeeping would be the wrong trade. ``clear`` is the
    explicit way to start over.
    """
    loop = state(thread_id) or {}
    loop.update({"on": True, "started_at": loop.get("started_at") or time.time()})
    if node_id:
        loop["node_id"] = str(node_id)
    if input_name:
        loop["input"] = str(input_name)
    loop.setdefault("node_id", "")
    loop.setdefault("input", "")
    loop.setdefault("versions", [])
    return _save(thread_id, loop) or loop


def stop(thread_id: str) -> dict | None:
    """Switch it off and keep the history — switching back on resumes it."""
    loop = state(thread_id)
    if not isinstance(loop, dict):
        return None
    loop["on"] = False
    return _save(thread_id, loop)


def clear(thread_id: str) -> None:
    """Forget the loop entirely (the only way a version list is thrown away)."""
    _save(thread_id, None)


def target(thread_id: str) -> tuple[str, str]:
    """``(node_id, input_name)`` the prompt goes to — either may be ""."""
    loop = state(thread_id) or {}
    return str(loop.get("node_id") or ""), str(loop.get("input") or "")


def set_target(thread_id: str, node_id: str, input_name: str = "") -> dict:
    loop = state(thread_id) or {"on": True, "versions": []}
    loop["node_id"] = str(node_id or "")
    if input_name:
        loop["input"] = str(input_name)
    return _save(thread_id, loop) or loop


def versions(thread_id: str) -> list:
    loop = state(thread_id) or {}
    out = loop.get("versions")
    return list(out) if isinstance(out, list) else []


def _active_of(loop: dict) -> dict | None:
    """The version on the canvas: the one picked in the strip, else the newest."""
    got = loop.get("versions") or []
    want = loop.get("active")
    for entry in got:
        if want and int(entry.get("v") or 0) == int(want):
            return entry
    return got[-1] if got else None


def current(thread_id: str) -> dict | None:
    """The version in force — the ACTIVE one, which is not always the newest.

    Picking v2 in the strip makes v2 active without writing a copy of it as v5:
    the list is what the agent has written, and which of those is on the canvas
    is a separate fact. The agent's next prompt is then based on v2.
    """
    loop = state(thread_id)
    return _active_of(loop) if isinstance(loop, dict) else None


def live_since(thread_id: str) -> float:
    """When the active version went onto the canvas — the floor for pairing a render.

    Its own ``at`` for a fresh version; the moment it was picked for an old one, or
    a render of whatever was on the canvas before the pick would be paired to it.
    """
    loop = state(thread_id) or {}
    entry = _active_of(loop) or {}
    return max(float(entry.get("at") or 0.0), float(loop.get("activated_at") or 0.0))


def activate(thread_id: str, v) -> dict | None:
    """Make version *v* the active one. Returns its entry, or None if there is none."""
    try:
        want = int(str(v).lstrip("vV"))
    except (TypeError, ValueError):
        return None
    loop = state(thread_id)
    if not isinstance(loop, dict):
        return None
    entry = next((e for e in loop.get("versions") or []
                  if int(e.get("v") or 0) == want), None)
    if entry is None:
        return None
    loop["active"] = want
    loop["activated_at"] = time.time()
    _save(thread_id, loop)
    return entry


def add_version(thread_id: str, text: str, *, node_id: str = "", input_name: str = "",
                based_on: int | None = None) -> dict:
    """Record *text* as the next version. Returns that version's entry."""
    loop = state(thread_id) or {"on": True, "versions": []}
    loop.setdefault("versions", [])
    if node_id:
        loop["node_id"] = str(node_id)
    if input_name:
        loop["input"] = str(input_name)
    entry = {
        "v": (max((int(e.get("v") or 0) for e in loop["versions"]), default=0) + 1),
        "text": str(text or ""),
        "at": time.time(),
        "output": "",
    }
    if not based_on:
        # Written while an OLDER version was active (picked in the strip): that is
        # what this one was made from, and the history should say so.
        live = _active_of(loop)
        if live and loop["versions"] and live is not loop["versions"][-1]:
            based_on = int(live.get("v") or 0)
    if based_on:
        entry["from"] = int(based_on)
    loop["versions"].append(entry)
    if len(loop["versions"]) > _MAX_VERSIONS:
        loop["versions"] = loop["versions"][-_MAX_VERSIONS:]
    loop["active"] = entry["v"]          # a new prompt is on the canvas: it is active
    loop.pop("activated_at", None)
    loop["on"] = True
    _save(thread_id, loop)
    return entry


def version_text(thread_id: str, v) -> str | None:
    """The text of version *v*, or None when there is no such version."""
    try:
        want = int(str(v).lstrip("vV"))
    except (TypeError, ValueError):
        return None
    for entry in versions(thread_id):
        if int(entry.get("v") or 0) == want:
            return str(entry.get("text") or "")
    return None


def pair_output(thread_id: str, path: str, v=None) -> dict | None:
    """Attach a render to the version that was live when it was made.

    With *v*: the panel queued that version and is reporting its job finished, so
    the render is v's even if another chip was clicked while it ran. Without: turn
    setup, pairing whatever ComfyUI produced most recently with the ACTIVE version
    — the text that was in the node when the graph was queued (the caller only
    passes renders newer than ``live_since``).
    Never overwrites: the first render a version gets is the one it is judged on, and
    a second queue of the same prompt is not a new fact about it.
    """
    loop = state(thread_id)
    if not isinstance(loop, dict) or not loop.get("versions") or not path:
        return None
    if v is None:
        live = _active_of(loop)
    else:
        want = str(v).lstrip("vV")
        live = next((e for e in loop["versions"] if str(e.get("v")) == want), None)
    if live is None or live.get("output"):
        return None
    live["output"] = str(path)
    _save(thread_id, loop)
    return live


def set_qa(thread_id: str, v, verdict: dict) -> dict | None:
    """Record the QA verdict on version *v*'s render.

    The panel queues the graph, so agentY's executor — where QA normally runs —
    never sees the run. The panel reports when its job finishes and the render is
    judged then; the verdict is kept on the version so it is judged once.
    """
    loop = state(thread_id)
    if not isinstance(loop, dict):
        return None
    for entry in loop.get("versions") or []:
        if str(entry.get("v")) == str(v).lstrip("vV"):
            entry["qa"] = dict(verdict)
            _save(thread_id, loop)
            return entry
    return None


def qa_retries(thread_id: str) -> int:
    """How many automatic QA retries this loop has made since the user last spoke."""
    loop = state(thread_id) or {}
    return int(loop.get("qa_retries") or 0)


def bump_qa_retries(thread_id: str) -> int:
    """Count one automatic retry; returns the new count."""
    loop = state(thread_id)
    if not isinstance(loop, dict):
        return 0
    loop["qa_retries"] = int(loop.get("qa_retries") or 0) + 1
    _save(thread_id, loop)
    return loop["qa_retries"]


def reset_qa_retries(thread_id: str) -> None:
    """Start the retry budget over — a pass, a spent budget, or the user steering."""
    loop = state(thread_id)
    if isinstance(loop, dict) and loop.get("qa_retries"):
        loop["qa_retries"] = 0
        _save(thread_id, loop)


def judge_version(thread_id: str, briefing, v=None, *, emit=None) -> dict | None:
    """Judge version *v*'s render (the active one when None) against *briefing*.

    Once per version: the verdict is stored on it, so the agent reads it in the
    loop block and a second call returns the stored one. *emit* receives the
    QA agent's work as a tool card — a call (what it judged) and a result (the
    verdict per criterion) — so it shows like every other agent's; the caller
    decides where cards go (the turn's stream, or the tool-activity buffer).
    Returns the verdict, or None when the version has no render to judge.
    """
    import json
    from src.utils.qa import check_output
    if v is None:
        entry = current(thread_id) or {}
    else:
        entry = next((e for e in versions(thread_id)
                      if str(e.get("v")) == str(v).lstrip("vV")), {})
    path = str(entry.get("output") or "")
    if not path:
        return None
    if isinstance(entry.get("qa"), dict):
        return entry["qa"]
    card = f"qa-loop-{str(thread_id)[:8]}-v{entry.get('v')}"
    describe = getattr(briefing, "describe", None)
    if emit is not None:
        emit({"phase": "call", "id": card, "agent": "qa", "name": "[qa] judge_render",
              "input": json.dumps({"version": entry.get("v"), "file": path,
                                   "briefing": describe() if callable(describe) else ""},
                                  ensure_ascii=False)})
    res = check_output(path, briefing, request=str(entry.get("text") or ""))
    verdict = {"passed": bool(res.passed), "summary": res.summary,
               "missed": res.failed_criteria(),
               "line": f"🔍 QA v{entry.get('v')} — {res.render()}"}
    if res.error or res.blind:
        verdict["error"] = res.error or "the QA model cannot read images"
    set_qa(thread_id, entry.get("v"), verdict)
    if emit is not None:
        checks = [f"{'✅' if str(c.get('result', '')).lower() in ('pass', 'n/a', 'na') else '❌'} "
                  f"{c.get('criterion', '')}" + (f" — {c['note']}" if c.get("note") else "")
                  for c in (res.checks or []) if isinstance(c, dict)]
        emit({"phase": "result", "id": card, "agent": "qa", "name": "[qa] judge_render",
              "result": "\n".join([res.render()] + checks)})
    return verdict


def _qa_line(verdict: dict) -> str:
    """One line of the block for a version's QA verdict."""
    if verdict.get("error"):
        return f"      QA: not judged ({verdict['error']})"
    if verdict.get("passed"):
        tail = f" — {verdict['summary']}" if verdict.get("summary") else ""
        return f"      QA: PASS{tail}"
    missed = "; ".join(verdict.get("missed") or []) or verdict.get("summary") or "?"
    return f"      QA: FAIL — missed: {missed}"


def newest_output(since: float = 0.0) -> str:
    """The newest image ComfyUI has finished writing, as a path, or "".

    The user queues the graph themselves, so nothing in agentY sees that run: its
    only trace is ComfyUI's own history. Best-effort by construction — a ComfyUI
    that is down, a history with no images, a file that cannot be resolved all mean
    "no render to look at", which is a fine thing for a turn to know.
    """
    try:
        from agenty_core.utils.comfyui_client import get_client
        history = get_client().get("/history", params={"max_items": 6})
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(history, dict):
        return ""

    def _stamp(entry) -> float:
        status = (entry or {}).get("status") or {}
        for name, payload in reversed(list(status.get("messages") or [])):
            if isinstance(payload, dict) and payload.get("timestamp"):
                try:                       # ComfyUI reports milliseconds
                    return float(payload["timestamp"]) / 1000.0
                except (TypeError, ValueError):
                    continue
        return 0.0

    candidates: list = []
    for entry in history.values():
        if not isinstance(entry, dict):
            continue
        when = _stamp(entry)
        if since and when and when < since:
            continue
        candidates += [(when, rec) for rec in _records(entry)]
    # Sort FIRST, resolve after. Resolving is a filesystem hit on whatever drive
    # ComfyUI writes to — a network share, here — and falls back to downloading the
    # file when it cannot find it, so resolving every record in six history entries
    # to then keep one is a stall on the front of somebody's turn.
    candidates.sort(key=lambda c: c[0], reverse=True)
    for _when, rec in candidates[:_RESOLVE_TRIES]:
        path = _resolve(rec)
        if path:
            return path
    return ""


def _records(entry: dict) -> list:
    """The saved-file records of one ComfyUI history entry (previews left out)."""
    out: list = []
    for node_out in ((entry or {}).get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "gifs", "videos"):
            for rec in (node_out.get(key) or []):
                if isinstance(rec, dict) and rec.get("type") != "temp":
                    out.append(rec)
    return out


def output_of(prompt_id: str) -> str:
    """The file one ComfyUI job wrote, as a path, or "".

    The panel knows the id of the job it queued, so this is exact where
    ``newest_output`` has to guess from timestamps.
    """
    if not prompt_id:
        return ""
    try:
        from agenty_core.utils.comfyui_client import get_client
        history = get_client().get(f"/history/{prompt_id}")
    except Exception:  # noqa: BLE001
        return ""
    entry = (history or {}).get(prompt_id) if isinstance(history, dict) else None
    for rec in _records(entry or {})[:_RESOLVE_TRIES]:
        path = _resolve(rec)
        if path:
            return path
    return ""


def _resolve(record: dict) -> str:
    """A ComfyUI output record as a path on disk, or "" when it cannot be found."""
    try:
        from src.executor import _resolve_output_path
        path = _resolve_output_path(str(record.get("filename") or ""),
                                    str(record.get("subfolder") or ""),
                                    str(record.get("type") or "output"))
        return str(path) if path and path.exists() else ""
    except Exception:  # noqa: BLE001
        return ""


def block(thread_id: str) -> str:
    """The loop's facts for the orchestrator's turn input, or "".

    The instructions live in ``config/system_prompts/orchestrator/prompt_loop.md``
    (a partial, like every other turn-conditional section); this is only what is
    true right now — where the prompt goes, what it says, and what the last render
    was. Newest version last, because that is the one being talked about.
    """
    loop = active(thread_id)
    if not loop:
        return ""
    node_id, input_name = str(loop.get("node_id") or ""), str(loop.get("input") or "")
    got = versions(thread_id)
    lines = []
    if node_id:
        lines.append(f"  Prompt target: node {node_id}"
                     + (f", input `{input_name}`" if input_name else ""))
    else:
        lines.append("  Prompt target: NOT SET — the first revise_prompt call must name "
                     "the node_id (the prompt node on their canvas; ask which one if the "
                     "graph has several and nothing is selected).")
    if not got:
        lines.append("  No prompt written yet in this loop.")
        return "\n".join(lines) + "\n"
    live = _active_of(loop) or got[-1]
    shown_versions = got[-_BLOCK_VERSIONS:]
    if live not in shown_versions:       # the active one is always shown
        shown_versions = [live] + shown_versions[1:]
    for entry in shown_versions:
        text = str(entry.get("text") or "")
        shown = text if len(text) <= _TEXT_IN_BLOCK else text[:_TEXT_IN_BLOCK] + " …"
        mark = "  v{v}{base}{now}: {text}".format(
            v=entry.get("v"), text=shown,
            base=f" (from v{entry['from']})" if entry.get("from") else "",
            now=" [ACTIVE, on the canvas now]" if entry is live else "")
        lines.append(mark)
        if entry.get("output"):
            lines.append(f"      rendered: {entry['output']}")
            if isinstance(entry.get("qa"), dict):
                lines.append(_qa_line(entry["qa"]))
    if len(got) > _BLOCK_VERSIONS:
        lines.insert(1, f"  ({len(got) - _BLOCK_VERSIONS} earlier version(s) not shown; "
                        f"ask for one by number if you need it.)")
    if live is not got[-1]:
        lines.append(f"  They picked v{live['v']} in the strip: it is the prompt on the "
                     f"canvas, and the one your next revision starts from — not "
                     f"v{got[-1]['v']}.")
    if live.get("output"):
        lines.append(f"  The render above is what v{live['v']} produced — it is what "
                     "they have just been looking at.")
    else:
        lines.append(f"  v{live['v']} has no render yet: it may still be running, or "
                     "it failed — ComfyUI's history does not show a finished image.")
    return "\n".join(lines) + "\n"
