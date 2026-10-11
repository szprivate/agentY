"""The run ledger: what has happened to each stage of a pipeline run.

A pipeline of hooks runs over many turns and, with branches, in several
conversations. Until now the only record of "the characters are approved, these
two files" was the agent's own memory of the conversation - and over a long run
it lost it: an approved screenplay was rewritten, a review long answered was
asked again, last round's pictures were handed on as the approved set.

So the host keeps the record, and the agent is told it every turn:

* it is written by the HOST, at the moments the host itself knows something -
  a stage produced files, a stop was raised, the user approved - never from
  what the agent says happened;
* it is kept with the LEAD conversation (conversation store), and the branches
  read and write that same copy;
* it holds paths, not content: a written result is saved as a versioned file
  and the ledger names the file.

One entry per stage (hook node), keyed by its node id::

    {"title", "purpose", "state", "round", "outputs": [...], "approved": [...],
     "text": "<path>", "review": "<id>", "by": "<branch>", "key": "<fingerprint>",
     "changed": bool, "updated": <time>}

States: ``pending`` (nothing yet), ``done`` (produced, nobody has to approve
it or nobody has yet), ``waiting`` (a review stop stands on it), ``approved``.
``changed`` is set beside ``done``/``approved`` when what the stage reads has
changed since it produced that result.
"""

from __future__ import annotations

import os
import re
import threading
import time

_KEY = "ledger"
_lock = threading.Lock()

PENDING, DONE, WAITING, APPROVED = "pending", "done", "waiting", "approved"
_INLINE_FILES = 4          # paths printed per stage in the agent's index


# ── the store ───────────────────────────────────────────────────────────────

def lead_of(thread_id: str) -> str:
    """The conversation a run's ledger is kept with: a branch's lead, else itself."""
    thread_id = str(thread_id or "")
    if not thread_id:
        return ""
    try:
        from src.utils import conversation_store as cs
        shot = cs.shot_of(thread_id)
        return str((shot or {}).get("lead_id") or thread_id)
    except Exception:  # noqa: BLE001
        return thread_id


def load(lead_id: str) -> dict:
    if not lead_id:
        return {"stages": {}, "order": []}
    from src.utils import conversation_store as cs
    got = cs.kv_get(lead_id, _KEY, None)
    if not isinstance(got, dict):
        got = {}
    got.setdefault("stages", {})
    got.setdefault("order", [])
    return got


def _save(lead_id: str, ledger: dict) -> None:
    from src.utils import conversation_store as cs
    cs.kv_set(lead_id, _KEY, ledger)


def _edit(lead_id: str, fn) -> dict:
    """Read, change with *fn(ledger)*, write - one writer at a time."""
    if not lead_id:
        return {"stages": {}, "order": []}
    with _lock:
        ledger = load(lead_id)
        fn(ledger)
        _save(lead_id, ledger)
    return ledger


def _stage(ledger: dict, hook_id: str) -> dict:
    hid = str(hook_id)
    st = ledger["stages"].setdefault(hid, {"state": PENDING, "round": 0})
    if hid not in ledger["order"]:
        ledger["order"].append(hid)
    return st


def _files(paths) -> list:
    return list(dict.fromkeys(str(p) for p in (paths or []) if str(p or "").strip()))


# ── what the host records ───────────────────────────────────────────────────

def sync(lead_id: str, hooks: list, keys: dict | None = None, partial: bool = False) -> dict:
    """Bring the ledger in line with the stages on the canvas.

    *hooks* as the turn planned them (in order), *keys* ``{hook_id: fingerprint}``
    of each as it stands now. A stage whose fingerprint differs from the one its
    result was made under is marked ``changed``. With *partial* (a branch, which
    sees its own stages only) nothing is said about the stages not listed.
    """
    keys = keys or {}

    def _do(ledger):
        seen = []
        for h in hooks or []:
            if not isinstance(h, dict) or h.get("hook_node_id") is None:
                continue
            hid = str(h["hook_node_id"])
            seen.append(hid)
            st = _stage(ledger, hid)
            st["title"] = _title(h)
            st["purpose"] = str(h.get("purpose") or "")
            st.pop("gone", None)
            now = keys.get(hid)
            if now and st.get("key") and st["state"] in (DONE, APPROVED):
                st["changed"] = bool(now != st["key"])
        if not partial:
            for hid, st in ledger["stages"].items():
                if hid not in seen:
                    st["gone"] = True
            ledger["order"] = seen + [h for h in ledger["order"] if h not in seen]
    return _edit(lead_id, _do)


def _title(hook: dict) -> str:
    title = " ".join(str(hook.get("title") or "").split())
    if title.lower() in ("", "agenty hook"):
        title = " ".join(str(hook.get("directive") or "").split())[:50]
    return title


def produced(lead_id: str, hook_id: str, outputs=None, text_path: str = "",
             key: str = "", by: str = "", new_round: bool = True) -> dict:
    """A stage made something. Returns its entry (``round`` is the one just made)."""
    out = {}

    def _do(ledger):
        st = _stage(ledger, hook_id)
        if new_round:
            st["round"] = int(st.get("round") or 0) + 1
            st["outputs"] = []
        st["outputs"] = _files(list(st.get("outputs") or []) + _files(outputs))
        if text_path:
            st["text"] = str(text_path)
        # Made again: whatever was approved was approved of the earlier round.
        st["state"] = DONE
        st.pop("approved", None)
        st.pop("review", None)
        st["changed"] = False
        if key:
            st["key"] = str(key)
        if by:
            st["by"] = str(by)
        st["updated"] = time.time()
        out.update(st)
    _edit(lead_id, _do)
    return out


def waiting(lead_id: str, review_id: str, stage_ids, files=None) -> None:
    """A review stop was raised on *stage_ids*; *files* are what is being asked about."""
    def _do(ledger):
        rv = _stage(ledger, review_id)
        rv["state"] = WAITING
        rv["outputs"] = _files(files)
        rv.pop("approved", None)
        rv["updated"] = time.time()
        for sid in stage_ids or []:
            st = _stage(ledger, sid)
            st["state"] = WAITING
            st["review"] = str(review_id)
            st["updated"] = time.time()
    _edit(lead_id, _do)


def approved(lead_id: str, review_id: str, stage_ids, files=None) -> None:
    """The user answered a review with a go-ahead; *files* are what goes on."""
    def _do(ledger):
        rv = _stage(ledger, review_id)
        rv["state"] = APPROVED
        rv["approved"] = _files(files)
        rv["updated"] = time.time()
        for sid in stage_ids or []:
            st = _stage(ledger, sid)
            st["state"] = APPROVED
            st["review"] = str(review_id)
            st["approved"] = _files(files)
            st["changed"] = False
            st["updated"] = time.time()
    _edit(lead_id, _do)


def reopened(lead_id: str, review_ids) -> None:
    """Reviews whose stage was redone, or a run that was stopped: they ask again."""
    def _do(ledger):
        for rid in review_ids or []:
            rv = ledger["stages"].get(str(rid))
            if rv is None:
                continue
            rv["state"] = PENDING
            rv.pop("approved", None)
            rv["updated"] = time.time()
            for st in ledger["stages"].values():
                if st.get("review") == str(rid) and st.get("state") in (APPROVED, WAITING):
                    st["state"] = DONE if st.get("outputs") or st.get("text") else PENDING
                    st.pop("approved", None)
    _edit(lead_id, _do)


# ── written results as files ────────────────────────────────────────────────

def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return s[:40] or fallback


def text_file(lead_id: str, hook_id: str, title: str, version: int, ctx: dict | None = None,
              root=None) -> str:
    """Where round *version* of a written stage is saved.

    In the project layout when a sequence/shot is in force
    (``<Seq>/<Shot>/text/<shot>_v003_<name>.md``, the same rule as the pictures),
    else with the agent's own notes (``agent/memory/texts/<run>/``).
    """
    name = _slug(title, f"hook_{hook_id}")
    v = f"v{int(version):03d}"
    try:
        from src.utils import output_context as _oc
        ctx = _oc.current() if ctx is None else ctx
        if root is None:
            root = _oc._output_root()
        if root is not None and _oc.active(ctx):
            stem = (ctx.get("shot") or ctx.get("sequence") or "output").lower()
            return os.path.join(str(root), *_oc.shot_root(ctx).split("/"), "text",
                                f"{stem}_{v}_{name}.md")
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.utils.hook_cache import cache_dir
        base = cache_dir(create=True)
    except Exception:  # noqa: BLE001
        base = None
    if base is None:
        return ""
    return os.path.join(str(base), "texts", _slug(lead_id, "run")[:16], f"{name}_{v}.md")


def write_text(path: str, text: str) -> bool:
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(str(text))
        return True
    except OSError:
        return False


# ── what the agent is told ──────────────────────────────────────────────────

_PURPOSE = {"set_parameter": "set / sweep parameter", "make_workflow": "make workflow",
            "text_only": "text only", "human_review": "review", "qa": "review (agent)",
            "loop_start": "loop start", "loop_break": "loop break", "join": "join"}


def _state_words(st: dict) -> str:
    state = st.get("state") or PENDING
    words = {PENDING: "not done yet", DONE: "done", WAITING: "WAITING FOR THE USER",
             APPROVED: "APPROVED by the user"}.get(state, state)
    if st.get("review") and state in (WAITING, APPROVED):
        words += f" (review {st['review']})"
    if st.get("changed") and state in (DONE, APPROVED):
        words += " — BUT ITS INPUTS HAVE CHANGED SINCE"
    return words


def _stage_line(hid: str, st: dict, full: bool = False) -> str:
    title = st.get("title") or ""
    purpose = _PURPOSE.get(st.get("purpose") or "", st.get("purpose") or "")
    head = f"- {hid}" + (f' "{title}"' if title else "") + (f" [{purpose}]" if purpose else "")
    bits = [_state_words(st)]
    if st.get("round"):
        bits.append(f"round {st['round']}")
    if st.get("by"):
        bits.append(f"by branch {st['by']}")
    line = head + ": " + " · ".join(bits)
    files = st.get("approved") if st.get("state") == APPROVED else st.get("outputs")
    files = list(files or [])
    what = "goes on" if st.get("state") == APPROVED else "made"
    if st.get("text"):
        line += f"\n    text → {st['text']}"
    if files:
        show = files if full else files[:_INLINE_FILES]
        line += f"\n    {what} ({len(files)}): " + "; ".join(show)
        if len(files) > len(show):
            line += f"; +{len(files) - len(show)} more (ledger_stage({hid}))"
    return line


def index(lead_id: str, only=None) -> str:
    """The block the agent is given each turn, or "" while nothing has happened."""
    ledger = load(lead_id)
    stages = ledger["stages"]
    ids = [h for h in ledger["order"] if h in stages and not stages[h].get("gone")]
    if only is not None:
        ids = [h for h in ids if h in {str(i) for i in only}]
    flow = {"loop_start", "loop_break", "join"}
    ids = [h for h in ids if (stages[h].get("purpose") or "") not in flow]
    if not any((stages[h].get("state") or PENDING) != PENDING for h in ids):
        return ""
    lines = [
        "[RUN LEDGER — kept by the host, not by you. This is what HAS happened in this "
        "pipeline run, across every turn and every branch, and it outranks your memory of "
        "the conversation. An APPROVED stage is settled: use exactly the files listed as "
        "what goes on, do not redo it, do not ask about it again. A stage whose inputs "
        "have changed since is NOT settled any more: say so and ask before relying on it. "
        "ledger_stage(<id>) gives a stage's full file list.]"]
    lines += [_stage_line(h, stages[h]) for h in ids]
    return "\n".join(lines)


def summary(lead_id: str) -> dict:
    """Counts for a one-line status: ``{"total", "approved", "waiting", "done"}``."""
    ledger = load(lead_id)
    work = [st for h, st in ledger["stages"].items()
            if not st.get("gone") and (st.get("purpose") or "") not in
            ("loop_start", "loop_break", "join", "human_review", "qa")]
    count = lambda s: sum(1 for st in work if st.get("state") == s)  # noqa: E731
    return {"total": len(work), "approved": count(APPROVED), "waiting": count(WAITING),
            "done": count(DONE)}


def stage(lead_id: str, hook_id: str) -> dict | None:
    st = load(lead_id)["stages"].get(str(hook_id))
    return dict(st) if st else None
