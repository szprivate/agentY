"""What a generated file is FOR, recorded where the next step will look.

An output leaves the executor as a path and arrives on the canvas as a loader
node named after its file. Everything the agent knew while making it — that these
five frames are the shots' start frames, that this one is the hero sheet — is
gone by the time anyone wires it into the next run, and the turn after that is
left describing pixels it has already described once.

Two records, answering two different questions:

* the **in-process registry** — what the CURRENT turn is producing. The panel
  reads it as each file lands, so the node it drops carries the role in its
  title instead of a filename. Per-turn, cleared with the turn.
* a **``.agenty.json`` sidecar** beside the file (and beside its staged copy in
  ComfyUI's input dir) — what a file is months later, in another thread. This is
  what a collector pointed at a folder of forty renders can read, and what makes
  an anchor render as "the hero sheet" rather than "LoadImage(hero_02.png)".

The role itself is never invented here. It comes from what the user wrote in the
hook's own prompt (``canvas_hooks.declared_output_role``), else the directive
that produced the file, else the brainbriefing — see the callers.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

SIDECAR_SUFFIX = ".agenty.json"
_MAX_ROLE = 120

_lock = threading.Lock()
_roles: dict[str, tuple] = {}    # resolved path -> (role, meta), frozen on first read
_run_role: str = ""              # the role for whatever the run in flight produces
_run_meta: dict = {}             # extra fields the sidecar should carry (hook, prompt)
_wf_roles: dict[str, tuple] = {}   # workflow path -> (role, meta) for what IT produces
_sources: dict[str, str] = {}      # output path -> the workflow that produced it
_wf_outputs: dict[str, list] = {}  # workflow path -> its outputs, in the order collected


def _key(path) -> str:
    try:
        return str(Path(str(path)).resolve()).lower()
    except Exception:  # noqa: BLE001
        return str(path or "").lower()


def clear() -> None:
    """Drop the per-turn registry (the sidecars on disk are the durable record)."""
    global _run_role, _run_meta
    with _lock:
        _roles.clear()
        _wf_roles.clear()
        _sources.clear()
        _wf_outputs.clear()
        _run_role = ""
        _run_meta = {}
    reset_dir_cache()


def set_run_role(role: str, **meta) -> None:
    """Declare what the run about to start is producing.

    Outputs appear asynchronously — the executor collects them while the server's
    pump is already emitting the earlier ones — so the role is set *before* the
    run rather than attached to each file afterwards, and each path freezes the
    role in force when it is first seen.
    """
    global _run_role, _run_meta
    with _lock:
        _run_role = " ".join(str(role or "").split())[:_MAX_ROLE]
        _run_meta = {k: v for k, v in meta.items() if v not in (None, "", [], {})}


def set_workflow_role(workflow_path, role: str, **meta) -> None:
    """Declare what one MEMBER of a batch produces, before it is submitted.

    A batch of five variants is five different things — five characters, five
    shots — and one role for the whole run cannot say which is which. The member
    is known by the workflow file it runs from, and that is known before anything
    executes, so by the time a file lands there is already a precise answer
    waiting for it. Doing this afterwards would be too late: the panel drops the
    node (and titles it) the moment the file appears.
    """
    r = " ".join(str(role or "").split())[:_MAX_ROLE]
    if not r:
        return
    with _lock:
        _wf_roles[_key(workflow_path)] = (r, {k: v for k, v in meta.items()
                                              if v not in (None, "", [], {})})


def note_source(output_path, workflow_path) -> None:
    """Record which workflow produced *output_path*, as the executor collects it.

    This is the join between "what was asked for" and "what came out". It is also
    what lets a batch report per-variant outputs at all: members are monitored
    concurrently and a healed one is re-queued, so the order files arrive in is
    not the order they were submitted in, and a flat list is a guess.
    """
    if not output_path or not workflow_path:
        return
    with _lock:
        _sources[_key(output_path)] = _key(workflow_path)
        _wf_outputs.setdefault(_key(workflow_path), []).append(str(output_path))


def outputs_of(workflow_path) -> list:
    """The outputs that workflow produced, in the order they were collected."""
    with _lock:
        return list(_wf_outputs.get(_key(workflow_path), ()))


def tag(path, role: str, **meta) -> None:
    """Record the role of one specific output, overriding the run's default."""
    r = " ".join(str(role or "").split())[:_MAX_ROLE]
    if not r:
        return
    with _lock:
        _roles[_key(path)] = (r, dict(meta))
    write_sidecar(path, r, **meta)


def _resolve(path) -> tuple[str, dict]:
    """The role and meta of *path*, frozen on first resolution.

    Frozen because the run's default moves on: a chained turn runs three stages,
    and the file produced by stage one must not pick up stage three's role just
    because the gallery got round to it late.
    """
    k = _key(path)
    with _lock:
        if k in _roles:
            return _roles[k]
        # The member that produced it knows better than the run it belonged to.
        found = _wf_roles.get(_sources.get(k, ""))
        role, meta = found if found else (_run_role, _run_meta)
        meta = dict(meta or {})
        if role:
            _roles[k] = (role, meta)
    if role:
        write_sidecar(path, role, **meta)
    return role, meta


def role_for(path) -> str:
    """The role of *path* — the run's, unless it was tagged individually."""
    return _resolve(path)[0]


def meta_for(path) -> dict:
    """What else was recorded about *path* (which hook, whether the user named it)."""
    return dict(_resolve(path)[1])


# ── the sidecar ───────────────────────────────────────────────────────────────

def sidecar_path(path) -> Path:
    """``<file>.agenty.json`` — beside the file, not in a database.

    A record that travels with the artifact survives a move, a copy into an input
    directory, a new thread, and a rebuilt project store. That is the whole point
    of it being a file.
    """
    return Path(str(path) + SIDECAR_SUFFIX)


def write_sidecar(path, role: str, **meta) -> bool:
    """Write (or refresh) the record beside *path*. Best-effort."""
    role = " ".join(str(role or "").split())[:_MAX_ROLE]
    if not role:
        return False
    try:
        p = Path(str(path))
        if not p.exists():
            return False
        body = {"role": role, "made_by": "agentY",
                "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
        for k, v in (meta or {}).items():
            if v not in (None, "", [], {}):
                body[k] = v
        sidecar_path(p).write_text(json.dumps(body, indent=1), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


def read_sidecar(path) -> dict:
    """The record beside *path*, or {} when there is none."""
    try:
        f = sidecar_path(path)
        if not f.is_file():
            return {}
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def role_of_file(path) -> str:
    """The recorded role of a file on disk — registry first, then its sidecar."""
    k = _key(path)
    with _lock:
        if k in _roles:
            return _roles[k][0]
    return str(read_sidecar(path).get("role") or "")


# ── resolving a loader's widget value to a real file ──────────────────────────
# A LoadImage on the canvas holds an input-relative filename, so the sidecar for
# it lives in ComfyUI's input dir. Resolving that costs a call to a ComfyUI that
# may not be running, so a miss is remembered for a while: this is a courtesy
# lookup on a hot path (every anchor of every hook, every turn), and it must
# never turn into a two-second stall per anchor.

_MISS_TTL = 20.0
_miss_until = 0.0


def input_dir() -> Path | None:
    """ComfyUI's input directory, or None when it can't be asked.

    A HIT is not cached here on purpose: ``get_comfyui_dirs`` already caches for
    the session and is reset when a session starts, which is how a project switch
    (a different input directory) is picked up without bookkeeping. Only the MISS
    is remembered, and briefly — a ComfyUI that is down costs two seconds per
    call, and this is asked once per anchor of every hook, every turn.
    """
    global _miss_until
    now = time.monotonic()
    if now < _miss_until:
        return None
    try:
        from src.tools.comfyui import get_comfyui_dirs  # lazy: avoid an import cycle
        d = (json.loads(get_comfyui_dirs()) or {}).get("input_dir") or ""
        p = Path(d) if d else None
        if p and p.is_dir():
            return p
    except Exception:  # noqa: BLE001
        pass
    _miss_until = now + _MISS_TTL
    return None


def role_of_canvas_file(name: str) -> str:
    """The role of a file a canvas node references by name (or absolute path)."""
    raw = str(name or "").strip().strip('"')
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return role_of_file(p)
    # ComfyUI subfolder syntax ("clipspace/x.png [input]") — take the path part.
    raw = raw.split(" [", 1)[0]
    base = input_dir()
    return role_of_file(base / raw) if base else ""


def reset_dir_cache() -> None:
    """Ask ComfyUI again on the next lookup, without waiting out the miss window."""
    global _miss_until
    _miss_until = 0.0
