"""What a conversation did, step by step - so it can be rebuilt as a hook pipeline.

"Turn what we just did into a pipeline" needs a faithful record of the
conversation: what the user asked, what came out, which workflow made each file,
what each step fed on, and how the user reacted. The agent's own memory is not
that record. Between turns its history is compacted, and an older conversation
is not in it at all. So it rebuilt from a summary, and guessed.

The record is here already, kept by the host:

* the transcript (``conversation_store.messages``): every user message and every
  reply, with its time;
* the gallery (``conversation_store.gallery``): every output of the conversation,
  in order, with its time;
* each output's provenance (``agenty_core.utils.provenance``): the exact graph
  that made it and the workflow file it ran from.

:func:`steps` lays them side by side. Each user message opens a step; the outputs
registered before the next user message are that step's. Each output gets the
workflow that made it, its main parameters (prompt, model, seed: the compact
form ``recreatable_params`` uses) and the earlier outputs of the conversation its
graph loaded, which is the wiring a pipeline needs. The user's next message is
the step's reaction: approval, a correction, another round.

The graph itself never goes into the result. When the workflow file a run named
is gone, the stored graph is written out as a workflow file of its own, so the
pipeline can still start from what worked.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MAX_STEPS = 40          # the newest steps of a long conversation
_ASK_CHARS = 400        # of each user message
_REPLY_CHARS = 240      # of the reply that ended the step
_PROMPT_CHARS = 200     # of a prompt text
_OUTPUTS_PER_STEP = 8   # files listed per step; the rest are counted


def _clip(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _stem(name: str) -> str:
    """A loader's value (``hero_00001_.png [output]``, ``sub/hero.png``) as a bare file name."""
    name = re.sub(r"\s*\[(input|output|temp)\]\s*$", "", str(name or "").strip())
    return os.path.basename(name.replace("\\", "/")).lower()


# ── the conversation ─────────────────────────────────────────────────────────

def resolve(conversation: str = "") -> tuple[str, str]:
    """``(thread_id, title)`` of the conversation meant: "" is the current one; else an
    id, else the best title match. ``("", why)`` when there is none."""
    from src.utils import conversation_store as cs
    want = str(conversation or "").strip()
    if not want:
        try:
            from src.utils.compaction import _current_thread
            want = _current_thread()
        except Exception:  # noqa: BLE001
            want = ""
        if not want:
            return "", "no conversation is open"
    got = cs.get_thread(want)
    if got:
        return str(got["id"]), str(got.get("title") or "")
    hits = cs.search_threads(want, limit=5)
    if not hits:
        return "", f"no conversation matches {want!r}"
    return str(hits[0]["id"]), str(hits[0].get("title") or "")


def _messages(thread_id: str) -> list[dict]:
    from src.utils import conversation_store as cs
    cs.init_db()
    with cs._connect() as conn:  # noqa: SLF001 - the store's own connection
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE thread_id=? ORDER BY id",
            (thread_id,)).fetchall()
    return [dict(r) for r in rows]


# ── one output ───────────────────────────────────────────────────────────────

def _loaded_names(graph: dict) -> set:
    """Every file name a graph's literal string inputs point at (loaders, mostly)."""
    names = set()
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, str) and "." in v and len(v) < 400 and "\n" not in v:
                names.add(_stem(v))
    return names


def _params(graph: dict) -> dict:
    """The few parameters that say what a run was: prompt, negative, model, seed."""
    out: dict = {}
    try:
        from agenty_core.tools.regenerate import param_index
        entries = param_index(graph)
    except Exception:  # noqa: BLE001
        entries = []
    for e in entries:
        names = set(e.get("names") or ()) | {str(e.get("label") or "")}
        value = e.get("value")
        for key in ("positive", "negative", "model", "seed"):
            if key in names and key not in out and value not in (None, ""):
                out[key] = {"id": e["id"],
                            "value": _clip(value, _PROMPT_CHARS) if isinstance(value, str) else value}
    return out


def _recovered_workflow(path: str, graph: dict) -> str:
    """Write the stored graph of *path* as a workflow file; its path, or ''."""
    try:
        from agenty_core.tools.comfyui import _workflows_dir
        folder = Path(_workflows_dir())
    except Exception:  # noqa: BLE001
        return ""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        from agenty_core.utils.provenance import canonical_sha1
        # Named by the graph, so the outputs of one run share one file.
        dest = folder / f"recovered_{canonical_sha1(graph)[:12]}.json"
        if not dest.is_file():
            dest.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(dest)
    except OSError:
        return ""


def describe_output(path: str) -> dict:
    """``{"file", "path", "workflow", "workflow_source", "made_with", "params", "loads"}``
    of one output. ``loads`` is the set of file names its graph read (internal)."""
    from agenty_core.utils import provenance as pv
    item = {"file": os.path.basename(str(path)), "path": str(path)}
    try:
        from agenty_core.utils.path_guard import refuse_read
        if refuse_read(path, "conversation_as_pipeline"):
            item["workflow_source"] = "not readable"
            return item
    except Exception:  # noqa: BLE001
        pass
    try:
        got = pv.provenance_of(path, use_history=False)
    except Exception:  # noqa: BLE001
        got = {"found": False}
    if not got.get("found") or not isinstance(got.get("graph"), dict):
        item["workflow_source"] = "none (not made by a ComfyUI run, or its record is gone)"
        return item
    graph = got["graph"]
    wf = str(got.get("workflow_file") or "")
    if wf and os.path.isfile(wf):
        item["workflow"], item["workflow_source"] = wf, "the workflow file that ran"
    else:
        rec = _recovered_workflow(path, graph)
        if rec:
            item["workflow"], item["workflow_source"] = rec, "rebuilt from the graph stored with the file"
    summary = pv.summarize(graph)
    made = [m["value"] for m in summary.get("models") or []][:3]
    made += [p["class"] for p in summary.get("partner") or []][:2]
    if made:
        item["made_with"] = made
    params = _params(graph)
    if params:
        item["params"] = params
    item["loads"] = _loaded_names(graph)
    return item


# ── the steps ────────────────────────────────────────────────────────────────

def steps(thread_id: str, *, max_steps: int = MAX_STEPS) -> dict:
    """The conversation *thread_id* as steps (see the module docstring)."""
    from src.utils import conversation_store as cs
    msgs = _messages(thread_id)
    gallery = cs.get_gallery(thread_id)
    asks = [m for m in msgs if m["role"] == "user" and str(m["content"]).strip()]
    if not asks:
        return {"steps": [], "workflows": [], "note": "the conversation has no messages from the user"}

    steps_: list[dict] = []
    for i, ask in enumerate(asks):
        start = ask["created_at"]
        end = asks[i + 1]["created_at"] if i + 1 < len(asks) else float("inf")
        reply = [m for m in msgs if m["role"] == "assistant" and start <= m["created_at"] < end]
        st = {"step": i + 1, "asked": _clip(ask["content"], _ASK_CHARS),
              "outputs": [g for g in gallery if start <= g["created_at"] < end]}
        if reply:
            st["replied"] = _clip(reply[-1]["content"], _REPLY_CHARS)
        if i + 1 < len(asks):
            st["reaction"] = _clip(asks[i + 1]["content"], _ASK_CHARS)
        steps_.append(st)
    # Outputs from before the first message (a restored gallery) go with step 1.
    early = [g for g in gallery if g["created_at"] < asks[0]["created_at"]]
    if early:
        steps_[0]["outputs"] = early + steps_[0]["outputs"]

    # Describe every output, oldest first, and find what each one read.
    by_name: dict = {}       # file name -> output number
    workflows: dict = {}     # workflow path -> {"steps", "outputs"}
    for st in steps_:
        described = []
        for g in st["outputs"]:
            item = describe_output(g["path"])
            item = {"n": int(g["idx"]), **item}
            loads = item.pop("loads", set())
            reads = sorted({by_name[n] for n in loads if n in by_name and by_name[n] != item["n"]})
            if reads:
                item["reads"] = reads
            by_name.setdefault(item["file"].lower(), item["n"])
            if item.get("workflow"):
                w = workflows.setdefault(item["workflow"], {"workflow": item["workflow"], "steps": [], "outputs": []})
                if st["step"] not in w["steps"]:
                    w["steps"].append(st["step"])
                w["outputs"].append(item["n"])
            described.append(item)
        st["outputs"] = described

    # What later steps built on, or what the last step made, is what "worked".
    read_later = {n for st in steps_ for o in st["outputs"] for n in o.get("reads", ())}
    last_made = next((st for st in reversed(steps_) if st["outputs"]), None)
    for st in steps_:
        for o in st["outputs"]:
            if o["n"] in read_later:
                o["used_by_a_later_step"] = True
        if st is last_made:
            st["last_step_with_outputs"] = True

    total = len(steps_)
    if total > max_steps:
        steps_ = steps_[-max_steps:]
    for st in steps_:
        if len(st["outputs"]) > _OUTPUTS_PER_STEP:
            more = len(st["outputs"]) - _OUTPUTS_PER_STEP
            kept = [o for o in st["outputs"] if o.get("used_by_a_later_step")][:_OUTPUTS_PER_STEP]
            kept += [o for o in st["outputs"] if o not in kept][: _OUTPUTS_PER_STEP - len(kept)]
            st["outputs"] = sorted(kept, key=lambda o: o["n"])
            st["more_outputs"] = more
    out = {"steps": steps_, "workflows": list(workflows.values())}
    if total > max_steps:
        out["note"] = f"the newest {max_steps} of {total} steps"
    return out


HOW_TO = (
    "Plan the pipeline from these steps with the hook-pipeline skill (read it first): "
    "section 'From a conversation that worked'. Leave out the detours (steps whose outputs "
    "nothing used and the user moved past), keep one stage per kind of work, start each "
    "generating stage from the workflow that made the result the user kept, put a human "
    "review where the user chose or corrected, and a loop where they asked for another "
    "round. Show the numbered plan; build it when they agree, or at once if they asked for it "
    "to be built.")
