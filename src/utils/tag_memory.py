"""Tags that outlive the graph they were written on.

An `agentY add tag` node names a wire: `#hero_face` means node 43, and node 43
only exists in the graph that is open. That is the right shape for pointing at an
input, and the wrong shape for the picture itself — the file is still there
tomorrow, in another workflow, in a Claude Desktop session on the same ComfyUI.

So the node carries a `remember` switch, and this module is what it does: write
the tag into the project's own memory as a named **reference** — the file's path
and what the user said it is for. From then on the name resolves in two places,
and :func:`resolve` is the order they are tried in: the canvas first, because a
tag on the graph in front of you is the more specific statement, then memory.

**Turning the switch off does not delete anything, and that is deliberate.** The
sync runs against whatever graph is open, and most graphs contain none of the
tags — so "forget what is not here" would empty the store the first time someone
opened an unrelated workflow. Off stops refreshing; removing is a gesture the
user makes in the project-memory editor, where they can see what they are
removing.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

# The first line is what the injected block shows for this entry, so it carries
# the meaning; the path is on its own line, labelled, for whatever reads it next.
_PATH_PREFIX = "path: "
# Says where the entry came from, in the file, for a human who opens it in a text
# editor and wonders why they never typed it.
_ORIGIN = "Remembered from the `agentY add tag` node tagged `#{tag}` on a canvas."


def stored_path(path: str) -> str:
    """The path as stored: FULL, wherever one can be worked out.

    This used to store the path input-RELATIVE when the file lived under
    ComfyUI's input directory, for portability between machines. That was the
    wrong trade. The entry is read by an agent that has to open the file;
    references routinely live outside the input dir (an output folder, a network
    share); and a relative name that no longer resolves is indistinguishable from
    one naming a different file. A full path is unambiguous on the machine that
    wrote it, which is the machine reading it back.

    A bare filename is resolved against the input dir — that is where ComfyUI
    means it — and anything that cannot be resolved is kept exactly as given
    rather than guessed at.
    """
    raw = str(path or "").strip().strip('"')
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        from agenty_core.tools.image_io import comfy_input_dir
        base = comfy_input_dir()
        if base:
            cand = Path(base) / raw
            if cand.is_file():
                return str(cand).replace("\\", "/")
    except Exception:  # noqa: BLE001
        pass
    return raw.replace("\\", "/")


# How far up the wire a tag's file is looked for. Bounded, and nearest-first, so
# a tag behind a couple of nodes still resolves without reaching across the graph
# and adopting some unrelated branch's image.
_FILE_SEARCH_HOPS = 6


def _file_upstream(base_prompt: dict, start_id: str) -> str:
    """The nearest file up the wire from *start_id*, or ''.

    Only the node the tag sits directly on used to be looked at, on the
    assumption that it is the loader. Often it is. But a tag wired behind a
    resize, a batch, a switch or an upscale names a node carrying no filename of
    its own, and the reference was then skipped SILENTLY: the tag stored nothing,
    and the agent — later asked to remember it — wrote its own prose describing
    the folder it could see instead of the file. Following the wire up costs
    nothing and is what the user meant by the tag either way.
    """
    seen, frontier, hops = {str(start_id)}, [str(start_id)], 0
    while frontier and hops <= _FILE_SEARCH_HOPS:
        nxt: list = []
        for nid in frontier:
            node = base_prompt.get(nid)
            if not isinstance(node, dict):
                continue
            hit = _file_of(node)
            if hit:
                return hit
            for value in (node.get("inputs") or {}).values():
                if isinstance(value, list) and value and str(value[0]) not in seen:
                    seen.add(str(value[0]))
                    nxt.append(str(value[0]))
        frontier, hops = nxt, hops + 1
    return ""


def _file_of(node: dict) -> str:
    """The media file a tagged node names, or ''.

    Only scalar widget values are considered — a link is a wire, not a file — and
    only ones that look like media, so a seed or a sampler name is never mistaken
    for a reference.
    """
    from src.utils.canvas_hooks import _looks_like_media_file

    for value in (node.get("inputs") or {}).values():
        if isinstance(value, str) and _looks_like_media_file(value):
            return value
    return ""


def entry_body(tag: str, file_path: str, role: str, origin: str = "") -> str:
    """What gets stored for one remembered tag."""
    head = str(role or "").strip() or f"Reference image `{Path(file_path).name}`."
    lines = [head]
    if file_path:
        lines.append(_PATH_PREFIX + stored_path(file_path))
    lines.append(origin or _ORIGIN.format(tag=tag))
    return "\n".join(lines)


_VIEWER_ORIGIN = "Tagged `#{tag}` for agentY from the bEpic viewer's history."

# Longest name a tag may be given. The name is also the entry's file name, and a
# name nobody can type back into a prompt is not a name.
_MAX_NAME = 64

# Every write to the store goes through this: the viewer, the Manage tags page and
# a turn's sync can all land at once, and a rename is a write plus a delete.
_store_lock = threading.RLock()


def valid_name(raw) -> str:
    """The tag *raw* names, normalised the way a tag node's field is, or ValueError.

    One rule for every way a name is given - the viewer's "Tag for agentY", a
    rename, a merge target - so a name that could be made can always be renamed
    to, and the other way round.
    """
    from src.utils.canvas_hooks import normalise_tag

    name = normalise_tag(raw)
    if not name or not any(c.isalpha() for c in name):
        raise ValueError("A tag needs a name with at least one letter, like hero_face.")
    if len(name) > _MAX_NAME:
        raise ValueError(f"A tag name can be at most {_MAX_NAME} characters long.")
    return name


def remember_file(tag: str, path: str, role: str = "") -> dict:
    """Tag one file from outside any graph - what the viewer's "Tag for agentY" does.

    The result is the entry an ``agentY add tag`` node with ``remember`` on would
    have written, so ``#tag`` resolves exactly as one does (:func:`resolve`): in a
    hook's prompt, in the collector's ``#`` menu, in what the agent is told is on
    record. Raises ``ValueError`` with a sentence for the user when it cannot.
    """
    from agenty_core.utils.project_memory import write_entry
    from src.utils.canvas_hooks import _looks_like_media_file, normalise_tag

    name = valid_name(tag)
    raw = str(path or "").strip().strip('"')
    if not raw or not Path(raw).is_file():
        raise ValueError("That picture is not on disk, so there is nothing to tag.")
    if not _looks_like_media_file(raw):
        raise ValueError("Only images, videos and other media files can be tagged.")
    role = " ".join(str(role or "").split())[:200]
    body = entry_body(name, raw, role, _VIEWER_ORIGIN.format(tag=name))
    with _store_lock:
        if write_entry(name, body, type="reference") is None:
            raise ValueError("The project memory could not be written - is ComfyUI running?")
    return {"tag": name, "path": stored_path(raw), "role": role}


def sync(base_prompt: dict | None) -> list[str]:
    """Write every remembered tag on this graph into project memory.

    Returns the tag names written. Safe to call on every turn: ``write_entry``
    replaces by name, so the same graph run a hundred times lands in the same
    file. A tag with its switch off, or one naming no file, is skipped — an entry
    whose whole content is "there was a tag here" is not a fact worth carrying.
    """
    if not isinstance(base_prompt, dict) or not base_prompt:
        return []
    try:
        from agenty_core.utils.project_memory import write_entry
        from src.utils.canvas_hooks import _REF_NOTE_CLASS, canvas_tags
    except Exception:  # noqa: BLE001
        return []

    remembered = {}
    for nid, node in base_prompt.items():
        if not isinstance(node, dict) or node.get("class_type") != _REF_NOTE_CLASS:
            continue
        if str((node.get("inputs") or {}).get("remember", "")).strip().lower() in \
                ("", "0", "false", "none", "no", "off"):
            continue
        remembered[str(nid)] = node

    if not remembered:
        return []

    written: list[str] = []
    for tag, info in canvas_tags(base_prompt).items():
        note = remembered.get(str(info.get("note_id")))
        if note is None:
            continue
        file_path = _file_upstream(base_prompt, str(info.get("node_id")))
        if not file_path:
            # Nothing up this wire names a file — a tag on a mid-graph tensor, or
            # a loader not pointed at anything yet. There is nothing to remember
            # but the name, and an entry whose whole content is "there was a tag
            # here" is not a fact worth carrying.
            continue
        try:
            if write_entry(tag, entry_body(tag, file_path, info.get("role") or ""),
                           type="reference") is not None:
                written.append(tag)
        except Exception:  # noqa: BLE001 — a convenience must never cost a turn
            continue
    return written


def remembered_reference(tag: str) -> dict | None:
    """A remembered tag by name: ``{"tag", "path", "role", "text"}`` or None.

    Read from the project store, so it answers in a graph that has never seen the
    tag node — which is the entire point of having written it.
    """
    from agenty_core.utils.project_memory import read_entry
    from src.utils.canvas_hooks import normalise_tag

    name = normalise_tag(tag)
    if not name:
        return None
    try:
        entry = read_entry(name)
    except Exception:  # noqa: BLE001
        return None
    if entry is None or str(getattr(entry, "type", "")) != "reference":
        return None
    text = str(getattr(entry, "body", "") or "")
    path, role = "", ""
    for i, line in enumerate(text.splitlines()):
        if line.startswith(_PATH_PREFIX):
            path = line[len(_PATH_PREFIX):].strip()
        elif i == 0:
            role = line.strip()
    return {"tag": name, "path": path, "role": role, "text": text}


def resolve(tag: str, canvas: dict | None) -> dict | None:
    """Where ``#tag`` points: the canvas first, then the project's memory.

    The canvas wins because a tag on the graph in front of you is the more
    specific statement — if both exist, the one you can see is the one you meant.
    Returns the ``canvas_tags`` shape with a ``source`` of ``"canvas"``, or the
    remembered entry with ``"memory"``, or None when the name means nothing.
    """
    from src.utils.canvas_hooks import canvas_tags, normalise_tag

    name = normalise_tag(tag)
    if not name:
        return None
    hit = (canvas_tags(canvas) or {}).get(name)
    if hit:
        return dict(hit, tag=name, source="canvas")
    remembered = remembered_reference(name)
    return dict(remembered, source="memory") if remembered else None


# ---------------------------------------------------------------------------
# Managing remembered tags (the Manage tags page, the viewer's "Remove tag")
# ---------------------------------------------------------------------------
# A remembered tag is one reference entry in the project store: one name, one
# file. The entry's file name is the name's slug ("hero_face" -> hero-face.md), so
# the name as the user wrote it is read back from the entry's own words: both
# origin lines quote it as `#name`. These functions are what the host's
# /agentY/tags routes call. They never touch a canvas: a tag on an `agentY add
# tag` node lives in the user's graph and is edited there, by the panel.
#
# Not journaled: the hook memory journal (hook_cache) records hook RESULTS, and a
# tag is not one. A renamed canvas tag does change the fingerprint of a hook that
# names it, which is right - the hook now reads a different name.

_MARK = re.compile(r"`#([A-Za-z0-9_\-]+)`")
_MAX_PATH = 4096


class TagError(ValueError):
    """A tag operation that cannot be done, with the HTTP status that says why."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _slug(name: str) -> str:
    from agenty_core.utils.project_memory import slug
    return slug(name)


def display_name(entry) -> str:
    """The name a remembered tag is written with in a prompt: ``hero_face``, not
    the file's ``hero-face``. Falls back to the file name for an entry no origin
    line names (one the agent wrote itself)."""
    found = ""
    for m in _MARK.finditer(str(getattr(entry, "body", "") or "")):
        if _slug(m.group(1)) == entry.name:
            found = m.group(1)
    return found or entry.name


def origin_of(body: str) -> str:
    """Who made a remembered tag: ``viewer``, ``canvas`` (a tag node's remember
    switch) or ``agent``."""
    text = str(body or "")
    if "from the bEpic viewer" in text:
        return "viewer"
    if "`agentY add tag` node tagged" in text:
        return "canvas"
    return "agent"


def _store_ready() -> None:
    from agenty_core.utils.project_memory import store_dir
    if store_dir() is None:
        raise TagError("The project memory is not reachable - is ComfyUI running?", 503)


def _find(tag):
    """The reference entry *tag* names, or TagError(404)."""
    from agenty_core.utils.project_memory import read_entry
    from src.utils.canvas_hooks import normalise_tag

    name = normalise_tag(tag)
    key = _slug(name)
    if not key or len(name) > 4 * _MAX_NAME:
        raise TagError("Which tag? That is not a tag name.")
    _store_ready()
    entry = read_entry(key)
    if entry is None or str(entry.type) != "reference" or not entry.file:
        raise TagError(f"There is no remembered tag #{name}.", 404)
    return entry


def _same_file(a: str, b: str) -> bool:
    def norm(p):
        return os.path.normcase(os.path.normpath(stored_path(p))) if str(p or "").strip() else ""
    na, nb = norm(a), norm(b)
    return bool(na) and na == nb


def _clean_path(path) -> str:
    raw = str(path or "").strip().strip('"')
    if not raw or len(raw) > _MAX_PATH or "\x00" in raw:
        raise TagError("That is not a file path.")
    return raw


def _retag_body(body: str, key: str, new: str) -> str:
    """The entry's words with every `#name` that meant this entry saying *new*."""
    hit = [False]

    def swap(m):
        if _slug(m.group(1)) != key:
            return m.group(0)
        hit[0] = True
        return f"`#{new}`"

    out = _MARK.sub(swap, str(body or ""))
    return out if hit[0] else out.rstrip() + f"\nTagged `#{new}`."


def _item(entry) -> dict:
    full = stored_path(entry.file)
    return {"tag": display_name(entry), "key": entry.name, "origin": origin_of(entry.body),
            "role": entry.summary, "path": full, "name": Path(full).name,
            "exists": Path(full).is_file()}


def list_tags() -> list[dict]:
    """Every remembered tag with its file, sorted by name.

    ``[{tag, key, origin, role, path, name, exists}]``. Only references that
    name a file are tags - the rest of the store is facts, and the `#` menu does
    not offer them either.
    """
    from agenty_core.utils.project_memory import list_entries

    _store_ready()
    out = [_item(e) for e in list_entries() if str(e.type) == "reference" and e.file]
    return sorted(out, key=lambda t: t["tag"].lower())


_RENAME_WARNING = ("Hook prompts and messages that say #{old} were not changed: "
                   "edit them to say #{new}.")


def _rename_locked(entry, new: str) -> dict:
    from agenty_core.utils.project_memory import delete_entry, read_entry, write_entry

    old = display_name(entry)
    key = _slug(new)
    if key != entry.name and read_entry(key) is not None:
        raise TagError(f"#{new} exists already. Merge the two tags instead.", 409)
    if write_entry(new, _retag_body(entry.body, entry.name, new), type=entry.type) is None:
        raise TagError("The project memory could not be written.", 503)
    if key != entry.name:
        delete_entry(entry.name)
    return {"tag": new, "from": old, "path": stored_path(entry.file),
            "warning": _RENAME_WARNING.format(old=old, new=new) if old != new else ""}


def rename_tag(tag, to) -> dict:
    """Give a remembered tag a new name: its entry moves, its file stays tagged."""
    new = valid_name(to)
    with _store_lock:
        return _rename_locked(_find(tag), new)


def delete_tag(tag) -> dict:
    """Forget a remembered tag. The picture itself is not touched."""
    from agenty_core.utils.project_memory import delete_entry

    with _store_lock:
        entry = _find(tag)
        if not delete_entry(entry.name):
            raise TagError("The project memory could not be written.", 503)
        return {"tag": display_name(entry), "path": stored_path(entry.file)}


def untag_file(tag, path) -> dict:
    """Take *tag* off one picture - the viewer's "Remove tag #name".

    A remembered tag names one file, so this forgets the tag when that file is
    the one given, and refuses (404) when the tag is on another picture: the
    viewer must not drop a tag from a picture the user is not looking at.
    """
    raw = _clean_path(path)
    with _store_lock:
        entry = _find(tag)
        if not _same_file(entry.file, raw):
            raise TagError(f"#{display_name(entry)} is not on that picture.", 404)
        return delete_tag(entry.name)


def merge_tags(source, into, keep: str = "into") -> dict:
    """Fold tag *source* into tag *into*; afterwards only *into* exists.

    A remembered tag holds one file, so when both have one, *keep* says whose
    picture the merged tag points at ("into", the default, or "from"); the
    other picture is no longer tagged, and comes back as ``dropped``. When
    *into* is not remembered yet this is a rename.
    """
    from agenty_core.utils.project_memory import delete_entry, read_entry, write_entry

    target = valid_name(into)
    if keep not in ("into", "from"):
        raise TagError('keep is "into" or "from".')
    with _store_lock:
        src = _find(source)
        key = _slug(target)
        if key == src.name:
            raise TagError("Those are the same tag.")
        dst = read_entry(key)
        if dst is None:
            out = _rename_locked(src, target)
            return dict(out, merged=True, dropped="")
        if str(dst.type) != "reference" or not dst.file:
            raise TagError(f"#{target} is a project fact, not a tag; it cannot be merged into.", 409)
        name = display_name(dst)
        if keep == "from":
            if write_entry(name, _retag_body(src.body, src.name, name), type="reference") is None:
                raise TagError("The project memory could not be written.", 503)
            kept, dropped = src.file, dst.file
        else:
            kept, dropped = dst.file, src.file
        delete_entry(src.name)
        old = display_name(src)
        return {"tag": name, "from": old, "path": stored_path(kept), "merged": True,
                "dropped": "" if _same_file(kept, dropped) else stored_path(dropped),
                "warning": _RENAME_WARNING.format(old=old, new=name)}
