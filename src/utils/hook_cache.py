"""Remember what a hook produced, so the same inputs don't buy the same answer twice.

A hook that reads an image and writes a description costs a vision call and a
turn of the agent's attention. Wire it into a graph you iterate on for an
afternoon and you pay for that same description twenty times, for a picture that
never changed. The hook node's switch says: keep what this produced, and while
nothing that feeds the hook has changed, put it straight back into the graph
without asking anyone.

**What counts as "what it produced" is everything.** Text and written prompts,
but also the scripts, images and videos a hook generated — media by *path*, since
the file already exists under the output directory and copying it would only
create a second truth. A remembered hook is replayed exactly the way a fresh one
delivers, so nothing downstream can tell the difference.

**The key is the question, not the answer.** It hashes the hook's own settings
(its prompt, purpose, where its output goes) together with the *upstream closure*
of everything wired into it — transitively, so a different image three nodes back
moves the key, and so does a rewire. Downstream is deliberately not in it:
changing where the value lands does not change what the value is.

"Wired into it" means **named as well as wired**: a reference a directive points
at with ``#hero_face`` is an input to the hook exactly like an anchor is, so it
is hashed like one — from the ``agentY add tag`` node itself, which pulls in both
the reference above it and the words on the tag. Without that, the one promise
this module makes ("released the moment anything feeding this hook changes")
would hold for a reference you wired and quietly fail for the identical
reference you named — you would swap the image and get the old answer back.

**The switch itself is deliberately NOT part of the key**, which is what makes it
usable in hindsight. You rarely know a result is worth keeping until you have
seen it, so what a hook produced is journalled whether or not the switch was on,
and turning it on afterwards lands on the key that run already wrote under. It is
still the forget gesture too: switch it off, send anything, switch it back on.

The store sits beside the outputs it describes — ``agent/memory/`` under
ComfyUI's own output directory, alongside ``agent/images`` and ``agent/videos``
— so a remembered path and the file it names travel together, and the whole lot
switches with the project. Paths are stored *relative* to that output directory
whenever they live under it, so moving the folder does not strand every entry.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_MAX_UPSTREAM = 400          # a runaway walk means a cyclic graph, not a big one
_FILE_KEYS = ("image", "video", "file", "path", "filename", "audio", "files",
              "lora_name", "ckpt_name")


# Where the store lives, under ComfyUI's OWN output directory — the same root
# that holds agent/images, agent/videos and agent/scripts. A remembered entry is
# mostly a set of paths into those folders, so keeping it anywhere else means the
# record and the files it names can be moved, backed up or deleted separately.
_STORE_PARTS = ("agent", "memory")

# What a remembered path is, by extension. Only used to label the record for the
# replay path and for anything that reads it back to a human.
_KINDS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"},
    "model": {".glb", ".gltf", ".obj", ".ply", ".stl"},
    "script": {".py", ".ps1", ".sh", ".bat"},
}

# How long a failed lookup is remembered. get_comfyui_dirs caches SUCCESS for the
# session but retries every failure, and a failure costs a two-second connection
# timeout — paid once per hook per turn without this.
_MISS_TTL = 20.0
_miss_until = 0.0


def output_dir() -> Path | None:
    """ComfyUI's output directory, as the running server reports it.

    None when ComfyUI is unreachable, which every caller here treats as "no
    memory this turn" rather than as an error. Success is deliberately not cached:
    the path is the project, and the day it changes under a running host is the
    day this has to notice.
    """
    global _miss_until
    now = time.monotonic()
    if now < _miss_until:
        return None
    try:
        from src.tools.comfyui import get_comfyui_dirs  # lazy: avoid an import cycle
        info = json.loads(get_comfyui_dirs()) or {}
    except Exception:  # noqa: BLE001
        _miss_until = now + _MISS_TTL
        return None
    raw = "" if not info or info.get("error") else str(info.get("output_dir") or "").strip()
    if not raw or raw == "unknown":
        _miss_until = now + _MISS_TTL
        return None
    try:
        return Path(raw)
    except Exception:  # noqa: BLE001
        return None


def forget_miss() -> None:
    """Drop the "ComfyUI didn't answer" note — for tests, and for a caller that
    knows the server just came up."""
    global _miss_until
    _miss_until = 0.0


def cache_dir(create: bool = False) -> Path | None:
    """``<output_dir>/agent/memory``, or None when ComfyUI can't be asked."""
    base = output_dir()
    if base is None:
        return None
    d = base.joinpath(*_STORE_PARTS)
    if create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return None
    return d


# ── the key ───────────────────────────────────────────────────────────────────

def _upstream(base_prompt: dict, start_ids) -> dict:
    """Every node feeding *start_ids*, transitively, keyed by id.

    Walks the API prompt's link form (``["12", 0]``), which is the same shape the
    executor runs — so what is hashed is what would actually be computed.
    """
    if not isinstance(base_prompt, dict):
        return {}
    out: dict = {}
    stack = [str(i) for i in start_ids if i is not None]
    while stack and len(out) < _MAX_UPSTREAM:
        nid = stack.pop()
        if nid in out or nid not in base_prompt:
            continue
        node = base_prompt[nid]
        if not isinstance(node, dict):
            continue
        out[nid] = node
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and val and val[0] is not None:
                stack.append(str(val[0]))
    return out


def _file_stamp(value) -> str:
    """``size:mtime`` for a file an input names, or ''.

    Names are not contents: ComfyUI's input directory is a place where ``ref.png``
    gets overwritten all afternoon. Without this the cache would happily answer a
    question about a picture that is no longer there.
    """
    raw = str(value or "").strip().strip('"').split(" [", 1)[0]
    if not raw or len(raw) > 400:
        return ""
    try:
        p = Path(raw)
        if not p.is_absolute():
            from src.utils.output_tags import input_dir
            base = input_dir()
            if base is None:
                return ""
            p = base / raw
        st = p.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except Exception:  # noqa: BLE001
        return ""


def _stamped(node: dict) -> dict:
    """A node's inputs, with any file they name replaced by name + size/mtime."""
    inputs = dict(node.get("inputs") or {})
    for key in _FILE_KEYS:
        val = inputs.get(key)
        if isinstance(val, str) and val.strip():
            stamp = _file_stamp(val) if key != "files" else ""
            if key == "files":  # a collector: one path per line
                stamps = [f"{ln.strip()}|{_file_stamp(ln)}"
                          for ln in val.splitlines() if ln.strip()]
                inputs[key] = "\n".join(stamps)
            elif stamp:
                inputs[key] = f"{val}|{stamp}"
    return {"class_type": node.get("class_type"), "inputs": inputs}


def _hook_identity(hook: dict) -> dict:
    """The hook's own settings — everything except whether it is remembering.

    The switch is left out on purpose (see the module docstring): off has to
    resolve to the key on wrote under, or turning it off could never release
    anything and turning it on afterwards could never find anything.

    That covers ``bake`` and ``freeze`` too, which used to be hashed here as two
    separate components and were never two things — the frontend derived one from
    the other (``freeze: bake``), so a single switch moved the key twice.
    """
    anchors = [
        {"node_id": str(a.get("node_id")), "to_input": a.get("to_input"),
         "slot": a.get("from_output_slot"), "role": a.get("role") or ""}
        for a in (hook.get("anchors") or []) if isinstance(a, dict)
    ]
    targets = [
        {"node_id": str(t.get("node_id")), "to_input": t.get("to_input")}
        for t in (hook.get("targets") or []) if isinstance(t, dict)
    ]
    return {
        "directive": str(hook.get("directive") or "").strip(),
        "purpose": str(hook.get("purpose") or "inline_parameter"),
        "anchors": sorted(anchors, key=lambda a: (a["node_id"], str(a["to_input"]))),
        "targets": sorted(targets, key=lambda t: (t["node_id"], str(t["to_input"]))),
        "prev_hook_ids": sorted(str(p) for p in (hook.get("prev_hook_ids") or [])),
    }


def _named_reference_ids(hook: dict, base_prompt: dict | None) -> list:
    """The ``agentY add tag`` nodes this hook's directive names with ``#tag``.

    Seeded from the TAG NODE rather than the reference it points at, because the
    walk from there goes up and so covers both in one: the note carries the tag
    and the stated role (change either and the agent is asked a different
    question), and above it sits the loader whose file is the reference itself.
    Starting at the loader would miss the note entirely — the note is downstream
    of it, and downstream is deliberately not in the key.
    """
    try:
        from src.utils.canvas_hooks import canvas_tags, mentioned_tags
    except Exception:  # noqa: BLE001 — a key must never cost the user a turn
        return []
    tags = canvas_tags(base_prompt)
    if not tags:
        return []
    out: list = []
    for name in mentioned_tags(hook.get("directive")):
        info = tags.get(name)
        nid = str((info or {}).get("note_id") or "")
        if nid and nid not in out:
            out.append(nid)
    return out


def fingerprint(hook: dict, base_prompt: dict | None) -> str:
    """The cache key for *hook* as it currently stands: its settings + its inputs."""
    anchor_ids = [str(a.get("node_id")) for a in (hook.get("anchors") or [])
                  if isinstance(a, dict) and a.get("node_id") is not None]
    if not anchor_ids and hook.get("anchor_node_id") is not None:
        anchor_ids = [str(hook["anchor_node_id"])]
    # A named reference is an input too — see the module docstring. Added to the
    # anchors rather than hashed separately so both kinds of input land in the same
    # upstream closure, and a reference that is BOTH wired and named is one entry.
    anchor_ids = anchor_ids + _named_reference_ids(hook, base_prompt)
    up = _upstream(base_prompt or {}, anchor_ids)
    payload = {
        "hook": _hook_identity(hook),
        "upstream": {nid: _stamped(node) for nid, node in sorted(up.items())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:24]


# ── the store ─────────────────────────────────────────────────────────────────

def _entry_path(key: str, create: bool = False) -> Path | None:
    d = cache_dir(create=create)
    if d is None or not str(key or "").strip():
        return None
    return d / f"{key}.json"


def read(key: str) -> dict | None:
    """The stored result for *key*, or None."""
    f = _entry_path(key)
    if f is None or not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def kind_of(path) -> str:
    """What a remembered file is — image / video / audio / model / script / file."""
    suffix = Path(str(path or "")).suffix.lower()
    for kind, exts in _KINDS.items():
        if suffix in exts:
            return kind
    return "file"


def _store_path(path) -> str:
    """A produced file as it is recorded: relative to the output dir when under it.

    The store lives inside that same directory, so a relative path keeps an entry
    valid when the whole folder is moved or the project is copied to another
    machine. Anything outside it is recorded absolute, because there is nothing
    sensible to make it relative to.
    """
    raw = str(path or "").strip().strip('"')
    if not raw:
        return ""
    base = output_dir()
    try:
        resolved = Path(raw).resolve()
    except Exception:  # noqa: BLE001
        return raw
    if base is not None:
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    return str(resolved)


def resolve_path(stored) -> Path | None:
    """The absolute path of a recorded output, or None when it can't be placed."""
    raw = str(stored or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    base = output_dir()
    return (base / raw) if base is not None else None


def _outputs_record(outputs) -> list:
    """Normalise produced files into the record's ``outputs`` list."""
    out: list = []
    seen: set = set()
    for item in (outputs or []):
        if isinstance(item, dict):
            path, role = item.get("path"), str(item.get("role") or "")
        else:
            path, role = item, ""
        stored = _store_path(path)
        if not stored or stored in seen:
            continue
        seen.add(stored)
        entry = {"path": stored, "kind": kind_of(stored)}
        if role:
            entry["role"] = role
        out.append(entry)
    return out


def write(key: str, value: str = "", outputs=None, **meta) -> bool:
    """Store what a hook produced under *key*.

    *value* is the text half (a written prompt, a description, a computed value)
    and *outputs* the files half (images, videos, scripts) as paths or
    ``{"path", "role"}`` dicts. Either may be empty — a hook that only wrote text
    and one that only produced images are both perfectly ordinary — but an entry
    with neither is not worth a file. Best-effort throughout: a store that cannot
    be written is not an error, it just means the work happens again next time.
    """
    files = _outputs_record(outputs)
    if not str(value or "").strip() and not files:
        return False
    f = _entry_path(key, create=True)
    if f is None:
        return False
    body: dict = {"when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if str(value or "").strip():
        body["value"] = str(value)
    if files:
        body["outputs"] = files
    for k, v in (meta or {}).items():
        if v not in (None, "", [], {}):
            body[k] = v
    try:
        f.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


def recall(key: str) -> dict | None:
    """A stored entry with its outputs resolved to absolute paths, or None.

    None when *any* remembered file has gone from disk. Deliberately all-or-
    nothing: a hook that produced five reference frames produced them as a set,
    and replaying four of them silently is a worse answer than doing the work
    again. So a tidied output folder degrades to a real run rather than to a
    graph pointing at files that aren't there.
    """
    entry = read(key)
    if entry is None:
        return None
    resolved: list = []
    for item in (entry.get("outputs") or []):
        if not isinstance(item, dict):
            continue
        path = resolve_path(item.get("path"))
        try:
            missing = path is None or not path.is_file()
        except OSError:
            missing = True
        if missing:
            return None
        out = dict(item)
        out["path"] = str(path)
        resolved.append(out)
    if not str(entry.get("value") or "").strip() and not resolved:
        return None
    out_entry = dict(entry)
    out_entry["outputs"] = resolved
    return out_entry


def kept(entry) -> bool:
    """Whether this entry was *blessed* — the switch was on for it — or only journalled.

    Every hook's result is written whether or not the switch was on, because you
    rarely know a result was worth keeping until you have looked at it. That makes
    the store two things at once: a short-lived journal of what just happened, and
    the durable memory of what someone chose to keep. Only the second survives
    pruning, and only the second is dropped by the forget gesture.
    """
    return bool((entry or {}).get("kept"))


def bless(key: str) -> bool:
    """Promote a journalled entry to a kept one. True when there was one to promote."""
    entry = read(key)
    if entry is None or kept(entry):
        return False
    entry["kept"] = True
    f = _entry_path(key, create=True)
    if f is None:
        return False
    try:
        f.write_text(json.dumps(entry, indent=1, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


# The journal is written on every turn for every hook, so it needs a bound. Kept
# entries are exempt — those are the user's answer to "was this worth keeping?"
# and are not something a housekeeping rule gets to overrule.
_JOURNAL_TTL_DAYS = 14
_JOURNAL_MAX = 200


def prune(ttl_days: int = _JOURNAL_TTL_DAYS, max_entries: int = _JOURNAL_MAX) -> int:
    """Drop journal-only entries that are stale or surplus. Returns how many went."""
    d = cache_dir()
    if d is None or not d.is_dir():
        return 0
    try:
        files = [f for f in d.glob("*.json") if f.is_file()]
    except OSError:
        return 0
    journal: list = []
    for f in files:
        entry = None
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass                       # unreadable: treat as journal, let it age out
        if kept(entry if isinstance(entry, dict) else None):
            continue
        try:
            journal.append((f.stat().st_mtime, f))
        except OSError:
            continue
    cutoff = time.time() - max(0, ttl_days) * 86400
    doomed = {f for mtime, f in journal if mtime < cutoff}
    survivors = sorted((m, f) for m, f in journal if f not in doomed)
    if len(survivors) > max(0, max_entries):
        doomed.update(f for _m, f in survivors[:len(survivors) - max_entries])
    gone = 0
    for f in doomed:
        try:
            f.unlink()
            gone += 1
        except OSError:
            continue
    return gone


def forget(key: str) -> bool:
    """Drop the entry for *key*. True when something was actually there."""
    f = _entry_path(key)
    if f is None or not f.is_file():
        return False
    try:
        f.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


# The make_workflow spellings, mirrored from canvas_hooks._STANDIN_PURPOSES. Only
# needed to read hooks from an older frontend, which sent two switches and used
# this purpose to decide which of them meant "keep it".
_STANDIN_PURPOSES = {"make_workflow", "make-workflow", "workflow-standin",
                     "workflow_standin", "standin", "workflow"}


def _on(val) -> bool:
    return val is True or str(val).strip().lower() in ("true", "1", "yes", "on")


def remembering(hook: dict) -> bool:
    """Whether this hook's keep-what-it-produced switch is on.

    One switch on the node, whose NAME follows the purpose — "bake into subgraph"
    on ``make_workflow``, "memorize result" everywhere else — but one bit on the
    wire, because they were never two questions: both answer "should what this
    produced outlive the run?".

    Older canvases sent two. Those are still read, resolved the way the node did
    at the time: ``bake`` was the switch ``make_workflow`` looked at, ``memorize``
    the one every other purpose looked at.
    """
    h = hook or {}
    if "remember" in h:
        return _on(h.get("remember"))
    purpose = str(h.get("purpose") or "").strip().lower()
    return _on(h.get("bake") if purpose in _STANDIN_PURPOSES else h.get("memorize"))


# The name the rest of the pipeline has always called this by.
memorizing = remembering
