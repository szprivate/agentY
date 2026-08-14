"""Remember what a hook produced, so the same inputs don't buy the same answer twice.

A hook that reads an image and writes a description costs a vision call and a
turn of the agent's attention. Wire it into a graph you iterate on for an
afternoon and you pay for that same description twenty times, for a picture that
never changed. The ``memorize`` toggle on the hook node says: keep the value, and
while nothing that feeds this hook has changed, put it straight back into the
graph without asking anyone.

**The key is the question, not the answer.** It hashes the hook's own settings
(its prompt, purpose, freeze, where its output goes) together with the *upstream
closure* of everything wired into it — transitively, so a different image three
nodes back moves the key, and so does a rewire. Downstream is deliberately not
in it: changing where the value lands does not change what the value is.

``memorize`` itself is deliberately NOT part of the key. Turning it off has to
*release* what was stored, which means the off state must resolve to the same key
the on state wrote under. That also makes the toggle the forget gesture: switch
it off, send anything, switch it back on.

The store sits beside the project — ComfyUI's user directory, the same place the
project memory lives — in its own ``cache/`` namespace, so it switches with the
project and never shows up in the memory block the agent reads every turn. A
cache entry is not something a human wants recited to them.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_MAX_UPSTREAM = 400          # a runaway walk means a cyclic graph, not a big one
_FILE_KEYS = ("image", "video", "file", "path", "filename", "audio", "files",
              "lora_name", "ckpt_name")


def cache_dir(create: bool = False) -> Path | None:
    """``<user_dir>/agentY/project/cache``, or None when there is no project store."""
    try:
        from src.utils.project_memory import user_dir
        base = user_dir()
    except Exception:  # noqa: BLE001
        return None
    if base is None:
        return None
    d = base / "agentY" / "project" / "cache"
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
    """The hook's own settings — everything except whether it is memorizing.

    ``memorize`` is left out on purpose (see the module docstring): the off state
    has to resolve to the key the on state wrote, or turning it off could never
    release anything.
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
        "freeze": bool(hook.get("freeze")),
        "bake": bool(hook.get("bake")),
        "anchors": sorted(anchors, key=lambda a: (a["node_id"], str(a["to_input"]))),
        "targets": sorted(targets, key=lambda t: (t["node_id"], str(t["to_input"]))),
        "prev_hook_ids": sorted(str(p) for p in (hook.get("prev_hook_ids") or [])),
    }


def fingerprint(hook: dict, base_prompt: dict | None) -> str:
    """The cache key for *hook* as it currently stands: its settings + its inputs."""
    anchor_ids = [str(a.get("node_id")) for a in (hook.get("anchors") or [])
                  if isinstance(a, dict) and a.get("node_id") is not None]
    if not anchor_ids and hook.get("anchor_node_id") is not None:
        anchor_ids = [str(hook["anchor_node_id"])]
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


def write(key: str, value: str, **meta) -> bool:
    """Store *value* under *key*. Best-effort; a cache that can't write is not an error."""
    f = _entry_path(key, create=True)
    if f is None or not str(value or "").strip():
        return False
    body = {"value": str(value), "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for k, v in (meta or {}).items():
        if v not in (None, "", [], {}):
            body[k] = v
    try:
        f.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


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


def memorizing(hook: dict) -> bool:
    """Whether this hook asked to be remembered (the node's ``memorize`` toggle)."""
    val = (hook or {}).get("memorize")
    return val is True or str(val).strip().lower() in ("true", "1", "yes", "on")
