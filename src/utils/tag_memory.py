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


def entry_body(tag: str, file_path: str, role: str) -> str:
    """What gets stored for one remembered tag."""
    head = str(role or "").strip() or f"Reference image `{Path(file_path).name}`."
    lines = [head]
    if file_path:
        lines.append(_PATH_PREFIX + stored_path(file_path))
    lines.append(_ORIGIN.format(tag=tag))
    return "\n".join(lines)


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
