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

# Inputs that say nothing a human would use to recognise a node. `control_after_
# generate` is UI bookkeeping, and the seed is both noisy and usually irrelevant
# to identifying which node this is.
_SKIP_INPUTS = {"control_after_generate"}

# Per-value and per-line limits. A prompt runs to paragraphs; showing the opening
# is enough to recognise it, and get_canvas_node returns the whole thing.
_VALUE_CHARS = 70
_LINE_CHARS = 190
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
    """One input's value, short enough to sit on a shared line."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return str(value)
    text = " ".join(str(value).split())
    return (text[:_VALUE_CHARS] + "…") if len(text) > _VALUE_CHARS else text


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
