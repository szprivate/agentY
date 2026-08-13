"""Per-project memory — the facts that are true of THIS project, stored beside it.

Where it lives is the whole idea. The store is a folder inside ComfyUI's **user
directory**, and the production pipeline (Ayon, here) already switches that
directory when it switches project, together with input and output. So the memory
follows the project without agentY holding a project id, a config entry, or a
switch of its own: ``get_comfyui_dirs()`` reports the running server's
``--user-directory`` straight out of its argv, and that result is cleared by
``clear_tool_caches()`` at the start of every session — which doubles, for free,
as this module's cache invalidation. Nothing here caches anything itself.

Shape on disk::

    <user_dir>/agentY/project/
      PROJECT.md                  generated index, for humans (never read back)
      technical/aspect-ratio.md
      character/hero.md
      style/grade.md
      reference/alley.md

One fact per file; the type is the folder, the name is the filename, and the
first non-empty line doubles as the summary the index shows. No frontmatter: these
files are meant to be opened in a text editor and synced by the pipeline, so the
format has to survive a human editing one by hand.

Deliberately NOT the FAISS store in ``src/utils/memory.py``. That one answers
"what have we learned, roughly, about this kind of task" by similarity, which is
the right model for lessons. This one answers "what is the hero's prompt" — by
name, exactly, every time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Folder layout under the ComfyUI user directory.
_STORE_PARTS = ("agentY", "project")
_INDEX_NAME = "PROJECT.md"

# The type is just the folder name, so anything is storable; these are the ones
# the agent is told about, in the order they read best in the injected block.
KNOWN_TYPES = ("technical", "character", "style", "reference", "note")
DEFAULT_TYPE = "note"

# Technical settings are short and apply to everything, so they go into the turn
# in full. Everything else is listed by name and pulled on demand — a project with
# forty characters must not cost forty character sheets' worth of context.
_ALWAYS_FULL = ("technical",)

# Caps for the injected block. The orchestrator's per-turn budget is the reason
# this store exists as files instead of one big document.
_MAX_LISTED = 40
_MAX_SUMMARY = 120
_MAX_FULL_CHARS = 1200


@dataclass(frozen=True)
class Entry:
    name: str      # slug, and the filename stem
    type: str      # folder it lives in
    body: str      # everything the file says
    path: Path

    @property
    def summary(self) -> str:
        """First non-empty line, which is what the index shows."""
        for line in self.body.splitlines():
            if line.strip():
                return line.strip()
        return ""


def slug(name: str) -> str:
    """Normalise a name to its filename form.

    Lookup goes through this on both sides, so "Hero", "hero" and "the hero"
    reach the same entry — an agent that writes a fact one turn and reads it back
    three turns later will not have kept the capitalisation.
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower())
    return s.strip("-")


def user_dir() -> Path | None:
    """ComfyUI's user directory, as the running server reports it.

    Returns None when ComfyUI is unreachable or reports nothing usable — every
    caller here treats that as "no project memory this turn", never as an error.
    """
    try:
        from src.tools.comfyui import get_comfyui_dirs  # lazy: avoid an import cycle
        info = json.loads(get_comfyui_dirs()) or {}
    except Exception:  # noqa: BLE001
        return None
    raw = str(info.get("user_dir") or "").strip()
    if not raw or raw == "unknown":
        return None
    try:
        return Path(raw)
    except Exception:  # noqa: BLE001
        return None


def store_dir(create: bool = False) -> Path | None:
    """The project store, or None when there is no ComfyUI to ask."""
    base = user_dir()
    if base is None:
        return None
    d = base.joinpath(*_STORE_PARTS)
    if create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return None
    return d


def list_entries() -> list[Entry]:
    """Every stored fact, technical first, then by type and name.

    Walks the folders rather than reading PROJECT.md: the files are the truth and
    the index is a rendering of them, so a hand-edited or stale index can never
    make a fact invisible.
    """
    d = store_dir()
    if d is None or not d.is_dir():
        return []
    out: list[Entry] = []
    for f in sorted(d.glob("*/*.md")):
        try:
            body = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:  # noqa: BLE001
            continue
        out.append(Entry(name=f.stem, type=f.parent.name, body=body, path=f))

    def order(e: Entry) -> tuple:
        known = list(KNOWN_TYPES)
        rank = known.index(e.type) if e.type in known else len(known)
        return (0 if e.type in _ALWAYS_FULL else 1, rank, e.name)

    return sorted(out, key=order)


def read_entry(name: str) -> Entry | None:
    """One fact by name, whatever type it was filed under."""
    want = slug(name)
    if not want:
        return None
    for e in list_entries():
        if e.name == want:
            return e
    return None


def write_entry(name: str, content: str, type: str = DEFAULT_TYPE) -> Entry | None:
    """Store one fact, replacing any entry of the same name.

    Replacing rather than appending is what makes this safe to call from a graph
    that runs a hundred times: the same key lands in the same file. A name that
    already exists under a DIFFERENT type moves, so re-filing a fact doesn't leave
    a second copy behind to contradict it later.
    """
    key, body = slug(name), str(content or "").strip()
    if not key or not body:
        return None
    d = store_dir(create=True)
    if d is None:
        return None
    folder = slug(type) or DEFAULT_TYPE
    existing = read_entry(key)
    target = d / folder / f"{key}.md"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body + "\n", encoding="utf-8")
        if existing is not None and existing.path != target:
            existing.path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        return None
    _write_index()
    return Entry(name=key, type=folder, body=body, path=target)


def delete_entry(name: str) -> bool:
    e = read_entry(name)
    if e is None:
        return False
    try:
        e.path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        return False
    _write_index()
    return True


def _write_index() -> None:
    """Regenerate PROJECT.md — a browsable view for whoever opens the folder.

    Never read back (see list_entries), so it cannot go stale in a way that
    matters; it exists because this directory is synced by the pipeline and a
    human will eventually look at it.
    """
    d = store_dir()
    if d is None or not d.is_dir():
        return
    lines = ["# Project memory", "",
             "Generated by agentY from the files in this folder. Edit the files, not this.", ""]
    for e in list_entries():
        lines.append(f"- **{e.name}** ({e.type}) — {e.summary}")
    try:
        (d / _INDEX_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_context() -> str:
    """The block injected into the turn, or '' when there is nothing to say.

    Technical settings arrive in full because they are short and bear on every
    decision; everything else arrives as one line so the agent knows what exists
    and can ask for the rest by name. That split is the whole token argument for
    keeping this out of the system prompt.
    """
    entries = list_entries()
    if not entries:
        return ""
    full = [e for e in entries if e.type in _ALWAYS_FULL]
    listed = [e for e in entries if e.type not in _ALWAYS_FULL][:_MAX_LISTED]

    out = ["[PROJECT MEMORY — facts established for THIS project, still in force.]"]
    if full:
        out.append("IN FORCE unless the user overrides them this turn:")
        for e in full:
            out.append(f"  - {e.name}: {_clip(e.body, _MAX_FULL_CHARS)}")
    if listed:
        out.append('ON RECORD — read one with project_memory_read("<name>") before you '
                   "invent your own version of it:")
        for e in listed:
            out.append(f"  - {e.name} ({e.type}) — {_clip(e.summary, _MAX_SUMMARY)}")
    return "\n".join(out)
