"""The whole graph the user has open, described so the agent can act on any of it.

Selection used to be the only way in: the agent saw the nodes the user had
selected and could write to exactly those. That made selecting a *permission*,
which is not what it is — it is a way of POINTING ("this one"), and needing it
for every edit turns "set the sampler to 30 steps" into "first go and click the
sampler". The graph is already sent on every turn (``graphToPrompt`` is what
ComfyUI runs on every Queue, so it costs nothing extra); it was simply never
described.

So the agent gets the whole canvas, and the selection keeps its real job: saying
which node the user means when they say "this one".

**Cost is the reason this is an index rather than a dump.** A real 20-node graph
is ~700 tokens as raw JSON and ~75 as one line per node; a large one would put
several thousand tokens into every turn, most of them about nodes nobody is
going to touch. So each node gets one line with its most useful values, values
are truncated, the block is capped, and anything past the cap is listed by id
and type alone. Whatever a line could not show in full is available exactly, on
request, through ``get_canvas_node`` — which is also why truncation here is safe.
"""

from __future__ import annotations

import os


def full_graph_visible() -> bool:
    """Whether the agent gets the whole canvas, or only what the user selected.

    Off by default, and the reason is cost rather than doctrine: the listing rides
    along on EVERY canvas turn, so it is paid for whether or not the turn was
    about the graph. Selection-only is the older behaviour and is free — the
    selection is usually a handful of nodes, and usually empty.

    ``AGENTY_CANVAS_FULL_GRAPH`` wins when set; otherwise ``canvas_full_graph``
    in settings.
    """
    env = os.environ.get("AGENTY_CANVAS_FULL_GRAPH")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("canvas_full_graph", False))
    except Exception:  # noqa: BLE001 — never let settings break a turn
        return False


# Inputs that say nothing a human would use to recognise a node. `control_after_
# generate` is UI bookkeeping, and the seed is both noisy and usually irrelevant
# to identifying which node this is.
_SKIP_INPUTS = {"control_after_generate"}

# Per-value and per-line limits. A prompt runs to paragraphs; showing the opening
# is enough to recognise it, and get_canvas_node returns the whole thing.
_VALUE_CHARS = 70
# A FILE PATH gets a far bigger budget than other values, because it is the one
# value on a line that the agent has to reproduce exactly rather than merely
# recognise, and because real ones are only a hundred-odd characters. Cut at 70 a
# loader path lost its filename and ended at a directory, and the agent wrote
# that directory into project memory as the reference. Middle-elision saved the
# filename but still leaves a `…` in the middle — recognisable, not usable — so
# the point is to not cut normal paths at all.
_PATH_CHARS = 160
# The whole line is capped too, and that cap has the same teeth: it cuts from the
# right, so a path sitting last on its line would lose its filename to THIS
# instead. Wide enough that a loader carrying a full path and a couple of small
# widgets fits. The block's own character budget still bounds the total.
_LINE_CHARS = 240
# Budget for the DESCRIBED lines, in characters (~4 chars/token). Past this,
# nodes are listed by id and type only — still findable, just not described.
_BLOCK_CHARS = 6000
# And a floor on that floor. Listing every node by id costs ~20 characters each,
# which is nothing at 20 nodes and a couple of thousand tokens at 400 — on every
# turn, mostly about nodes nobody will touch. Past this many, the remainder is
# counted rather than named: a graph that big is not one the agent should be
# scanning by eye anyway, and get_canvas_node still reaches any of them.
_MAX_LISTED = 250


def _title(node: dict) -> str:
    """The user's own title for a node, or "" when it carries the default."""
    meta = node.get("_meta") if isinstance(node, dict) else None
    title = str((meta or {}).get("title") or "").strip()
    cls = str((node or {}).get("class_type") or "").strip()
    return "" if not title or title == cls else title[:60]


def _render_value(value) -> str:
    """One input's value, short enough to sit on a shared line.

    A PATH is shortened in the middle rather than at the end. Cutting a path from
    the right removes its filename — the one part that says which file it is —
    and what survives ends at a directory, which is then read as one. That is not
    hypothetical: shown ``…\\RND_0500\\ima…`` for a loader, the agent reported the
    reference as "a directory, not a specific image file" and wrote it into
    project memory that way. Prose is still cut from the end, where the opening
    is what identifies it.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    text = " ".join(str(value).split())
    if len(text) <= _VALUE_CHARS:
        return text
    # A separator alone does not make it a path — "a 50/50 split composition" is
    # prose, and keeping its last clause instead of its opening helps nobody. A
    # FILENAME is wanted, so the last segment has to look like one: an extension,
    # and short enough that keeping it still shortens the value.
    tail = text.replace("\\", "/").rsplit("/", 1)[-1]
    if tail != text and "." in tail and 0 < len(tail) <= _PATH_CHARS - 8:
        if len(text) <= _PATH_CHARS:
            return text                       # shown whole: usable, not just legible
        return text[:_PATH_CHARS - len(tail) - 2] + "…/" + tail
    return text[:_VALUE_CHARS] + "…"


def node_line(node_id: str, node: dict, *, values: bool = True) -> str:
    """``#12 KSampler "Main sampler" — steps=30, cfg=6.5``.

    Link inputs are left out: they are wiring, not parameters, and the agent
    cannot set them with ``set_canvas_node_params`` anyway.
    """
    cls = str((node or {}).get("class_type") or "?")
    title = _title(node)
    head = f'#{node_id} {cls}' + (f' "{title}"' if title else "")
    if not values:
        return head
    parts = []
    for name, value in ((node or {}).get("inputs") or {}).items():
        if isinstance(value, (list, dict)) or name in _SKIP_INPUTS:
            continue      # a wire, or bookkeeping
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        parts.append(f"{name}={_render_value(value)}")
    if not parts:
        return head
    line = f"{head} — " + ", ".join(parts)
    return (line[:_LINE_CHARS] + "…") if len(line) > _LINE_CHARS else line


def describe_canvas(prompt: dict | None, selected_ids=None) -> str:
    """The ``[CANVAS GRAPH]`` block: every node the user has open, one per line.

    Nodes the user has SELECTED are marked, because that is what a selection now
    means — "the one I am pointing at" — and it is the only thing the agent
    cannot work out for itself from the graph.
    """
    if not isinstance(prompt, dict) or not prompt:
        return ""
    selected = {str(s) for s in (selected_ids or [])}
    lines, over, used, unlisted = [], [], 0, 0
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if used >= _BLOCK_CHARS:
            if len(lines) + len(over) >= _MAX_LISTED:
                unlisted += 1
            else:
                over.append(node_line(str(nid), node, values=False))
            continue
        line = node_line(str(nid), node)
        if str(nid) in selected:
            line += "   ← SELECTED"
        used += len(line) + 1
        lines.append(line)
    if not lines and not over:
        return ""

    head = (
        f"[CANVAS GRAPH — the {len(lines) + len(over) + unlisted} node(s) on the workflow the "
        "user has open, as it stands right now. You can read any of them here and "
        "change any of them with `set_canvas_node_params(node_id, {...})` — a node "
        "does NOT have to be selected. Selection is how the user POINTS at a node "
        "(\"this one\"), not permission to touch it.\n"
        "  Values are shortened to fit; a `…` means there is more. Call "
        "`get_canvas_node(node_id)` for a node's exact values before you rewrite "
        "one — never edit a value you have only seen truncated.\n"
        "  Wired inputs are not listed: those are links, and set_canvas_node_params "
        "writes widget values, not wiring. Editing here does NOT queue the graph — "
        "the user runs it themselves.]"
    )
    body = "\n".join(lines)
    tail = ""
    if over:
        tail = ("\n  (too many nodes to describe in full — the rest, by id and type "
                "only; use get_canvas_node for any of them)\n" + "\n".join(over))
    if unlisted:
        tail += (f"\n  (+{unlisted} more node(s) not listed — this graph is too large "
                 "to put in front of you every turn. get_canvas_node still reaches "
                 "any node by id; ask the user which one they mean rather than "
                 "guessing.)")
    return f"{head}\n{body}{tail}"


def deletion_impact(prompt: dict | None, node_ids) -> dict:
    """What deleting *node_ids* removes, and what it breaks downstream.

    Deleting is the one canvas edit that destroys something, so the answer has to
    be readable BEFORE it happens: which nodes these actually are (an id alone is
    not a thing anyone can picture), and which inputs elsewhere lose their feed —
    a graph that no longer runs because an input silently emptied is a worse
    outcome than the node still being there.

    Returns ``{"found": [...], "missing": [...], "orphaned": [...]}``.
    """
    graph = prompt if isinstance(prompt, dict) else {}
    wanted = [str(n) for n in (node_ids or [])]
    doomed = {n for n in wanted if n in graph}
    found = [{"node_id": n, "class_type": str((graph.get(n) or {}).get("class_type") or ""),
              "title": _title(graph.get(n) or {})} for n in wanted if n in doomed]
    orphaned = []
    for nid, node in graph.items():
        if str(nid) in doomed or not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in doomed:
                orphaned.append({
                    "node_id": str(nid),
                    "class_type": str(node.get("class_type") or ""),
                    "input": str(name),
                    "was_fed_by": str(value[0]),
                })
    return {"found": found,
            "missing": [n for n in wanted if n not in doomed],
            "orphaned": orphaned}


def node_detail(prompt: dict | None, node_id: str) -> dict | None:
    """Everything one node carries: its class, title, values and wired inputs.

    Untruncated on purpose — this is what a line in the block could not show, and
    the reason a truncated line is safe to print in the first place.
    """
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        return None
    values, wired = {}, {}
    for name, value in (node.get("inputs") or {}).items():
        if isinstance(value, list) and len(value) == 2:
            wired[name] = f"from #{value[0]} output {value[1]}"
        else:
            values[name] = value
    return {
        "node_id": str(node_id),
        "class_type": str(node.get("class_type") or ""),
        "title": _title(node),
        "values": values,
        "wired_inputs": wired,
    }
