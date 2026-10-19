"""Propose the chunks of a canvas graph, for groups a person can read.

The agent can already group the graphs it BUILDS (agenty_core's graph_groups,
run while a workflow is assembled). A graph the user built, or one the agent
grew by hand, has none of that: this proposes the same kind of chunks from the
graph as it stands on the canvas, so ``auto_group_canvas`` can draw them.

Pure and host-free: the API-format graph and a ``schemas`` object with
``.get(class_type)`` (``/object_info`` for one class) in, plain dicts out. Where
the boxes go and how big the nodes are is the panel's business - it is the only
one that knows - so a proposal is node ids, a title and a colour, nothing more.

How a node is sorted (by what the schema says; class names only where the schema
carries no signal):

* ``hooks``    - agentY's own pipeline nodes (``AgentY*``)
* ``finish``   - an output node (Save*, Preview*, anything ``output_node``) and
                 the VAE decode / encode beside it
* ``sampling`` - handed a ``MODEL`` (every sampler and guider)
* ``prompt``   - emits ``CONDITIONING``, or is a bare text source
* ``models``   - a root that supplies a model, clip, vae, controlnet, lora...
* ``inputs``   - a root that supplies media (LoadImage, LoadVideo, ...)
* anything else takes the role of what it feeds, else of what feeds it.

With two or more generating stages the chunks are cut per stage (graph_groups'
hand-off rule: a stage ends where an IMAGE/VIDEO/AUDIO wire starts) and titled
``<stage> · <role>``.
"""
from __future__ import annotations

import re

# One colour per role, the same on every graph: the colour says what a box is
# before you read it. Hex values from ComfyUI's own group-colour menu.
ROLE_COLORS = {
    "models": "#3f789e",    # blue
    "inputs": "#88A",       # slate
    "prompt": "#b58b2a",    # yellow
    "sampling": "#a1309b",  # purple
    "finish": "#3f9e6b",    # green
    "hooks": "#b06634",     # orange
    "other": "#8A8",        # grey-green
}
ROLE_LABELS = {
    "models": "Load models",
    "inputs": "Input media",
    "prompt": "Prompt",
    "sampling": "Sampling",
    "finish": "Decode & save",
    "hooks": "Hooks",
    "other": "Processing",
}
# Reading order: what feeds comes before what it feeds.
ROLE_ORDER = ("models", "inputs", "prompt", "sampling", "other", "finish", "hooks")

_MEDIA_OUT = {"IMAGE", "VIDEO", "AUDIO", "MASK"}
_MODEL_OUT = {"MODEL", "CLIP", "VAE", "CLIP_VISION", "CONTROL_NET", "UPSCALE_MODEL", "STYLE_MODEL",
              "GLIGEN", "LORA_MODEL", "AUDIO_ENCODER", "LATENT_UPSCALE_MODEL"}


def _is_link(v) -> bool:
    return (isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int))
            and not isinstance(v[0], bool) and isinstance(v[1], int))


def _inputs_of(schema: dict) -> dict:
    block = (schema or {}).get("input") or {}
    out = {}
    for kind in ("required", "optional"):
        for name, spec in (block.get(kind) or {}).items():
            out[name] = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    return out


def _out_types(schema: dict) -> list:
    return [str(t) for t in ((schema or {}).get("output") or []) if isinstance(t, str)]


def _key(n):
    return (0, int(n), "") if str(n).isdigit() else (1, 0, str(n))


def levels(graph: dict) -> dict:
    """Longest-path depth of every node (roots are 0); cycle-safe."""
    parents = {k: [str(v[0]) for v in (n.get("inputs") or {}).values()
                   if _is_link(v) and str(v[0]) in graph] for k, n in graph.items()}
    level = {k: 0 for k in graph}
    for _ in range(len(graph) + 1):
        moved = False
        for k, ps in parents.items():
            want = max((level[p] + 1 for p in ps), default=0)
            if want > level[k] and want <= len(graph):
                level[k] = want
                moved = True
        if not moved:
            break
    return level


def classify(graph: dict, schemas) -> dict:
    """``{node_id: role}`` for every node of *graph* (see the module docstring)."""
    role: dict = {}
    parents = {k: [] for k in graph}
    children = {k: [] for k in graph}
    for k, n in graph.items():
        for v in (n.get("inputs") or {}).values():
            if _is_link(v) and str(v[0]) in graph:
                parents[k].append(str(v[0]))
                children[str(v[0])].append(k)
    for k, n in graph.items():
        cls = str(n.get("class_type") or "")
        schema = schemas.get(cls) or {}
        ins = _inputs_of(schema)
        outs = set(_out_types(schema))
        root = not any(_is_link(v) for v in (n.get("inputs") or {}).values())
        in_types = {str(t) for t in ins.values() if isinstance(t, str)}
        if cls.startswith("AgentY"):
            role[k] = "hooks"
        elif schema.get("output_node") or re.match(r"^(Save|Preview)", cls):
            role[k] = "finish"
        elif "MODEL" in in_types or re.search(r"Sampler|Guider", cls):
            role[k] = "sampling"
        elif "CONDITIONING" in outs or "TextEncode" in cls:
            role[k] = "prompt"
        elif root and (outs & _MODEL_OUT or (re.search(r"Loader$|^Load[A-Z]", cls) and not outs & _MEDIA_OUT)):
            role[k] = "models"
        elif root and outs & _MEDIA_OUT:
            role[k] = "inputs"
        elif root and outs and outs <= {"STRING"}:
            role[k] = "prompt"
        elif re.search(r"VAEDecode|VAEEncode", cls):
            role[k] = "finish"
    # What is still unsorted follows the nearest sorted neighbour: what it feeds
    # first (a helper belongs with its user), then what feeds it.
    lv = levels(graph)
    for _ in range(len(graph)):
        left = [k for k in graph if k not in role]
        if not left:
            break
        progress = False
        for k in sorted(left, key=lambda x: (-lv.get(x, 0), _key(x))):
            near = [role[c] for c in children[k] if c in role] or [role[p] for p in parents[k] if p in role]
            if near:
                role[k] = max(sorted(set(near), key=ROLE_ORDER.index), key=near.count)
                progress = True
        if not progress:
            break
    for k in graph:
        role.setdefault(k, "other")
    return role


def _meta(graph: dict, schemas) -> dict:
    meta = {}
    for k, n in graph.items():
        schema = schemas.get(str(n.get("class_type") or "")) or {}
        ins = _inputs_of(schema)
        conns = [(name, ins[name] if isinstance(ins.get(name), str) else "*")
                 for name, v in (n.get("inputs") or {}).items() if _is_link(v)]
        meta[k] = {"conns": conns, "outputs": [(f"o{i}", t) for i, t in enumerate(_out_types(schema))]}
    return meta


class _Info:
    def __init__(self, schemas):
        self._schemas = schemas

    def get(self, cls, default=None):
        return self._schemas.get(cls) or default


def _stages(graph: dict, schemas, lv: dict) -> tuple:
    """``({node: stage anchor}, {anchor: title})`` - empty unless the graph has
    two or more generators, by graph_groups' own rule."""
    try:
        from agenty_core.tools import graph_groups as gg  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - without it the graph is one stage
        return {}, {}
    meta, info = _meta(graph, schemas), _Info(schemas)
    anchors = gg.find_anchors(graph, meta, info)
    if len(anchors) < 2:
        return {}, {}
    stage = gg.assign_stages(graph, meta, anchors, lv)
    titles = {}
    for owner in set(stage.values()):
        if owner == gg.SHARED:
            titles[owner] = "Shared"
            continue
        members = [k for k, o in stage.items() if o == owner]
        titles[owner] = gg._stage_title(graph, meta, members, owner, info)  # noqa: SLF001
    return stage, titles


def propose(graph: dict, schemas, min_nodes: int = 4) -> list:
    """The chunks of *graph*: ``[{title, role, color, node_ids}]``, reading order.

    ``[]`` for a graph too small to be worth boxing."""
    graph = {str(k): v for k, v in (graph or {}).items() if isinstance(v, dict)}
    if len(graph) < min_nodes:
        return []
    lv = levels(graph)
    role = classify(graph, schemas)
    stage, stage_title = _stages(graph, schemas, lv)
    chunks: dict = {}
    for k in graph:
        chunks.setdefault((stage.get(k, ""), role[k]), []).append(k)

    def order(item):
        (st, r), ids = item
        return (min(lv[i] for i in ids) if st else 0, ROLE_ORDER.index(r), str(st))

    out = []
    for (st, r), ids in sorted(chunks.items(), key=order):
        label = ROLE_LABELS[r]
        title = f"{stage_title.get(st, 'Shared')} · {label}" if stage else label
        out.append({"title": title, "role": r, "color": ROLE_COLORS[r],
                    "node_ids": sorted(ids, key=_key)})
    # A single chunk says nothing a frame around the whole graph would not.
    return out if len(out) > 1 else []


def clean_group_specs(groups) -> tuple:
    """``(specs, errors)`` from what the model sent to ``group_canvas_nodes``."""
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, list) or not groups:
        return [], ["groups must be a non-empty list of {title, node_ids, color?, rename_from?}."]
    specs, errors, seen = [], [], set()
    for i, g in enumerate(groups, start=1):
        if not isinstance(g, dict):
            errors.append(f"group {i} is not an object.")
            continue
        title = str(g.get("title") or "").strip()
        ids = [str(n).strip().lstrip("#") for n in (g.get("node_ids") or []) if str(n).strip()]
        ids = list(dict.fromkeys(ids))
        rename = str(g.get("rename_from") or "").strip()
        color = str(g.get("color") or "").strip()
        name = title or f"#{i}"
        if not title:
            errors.append(f"group {i} has no title.")
        if not ids and not rename:
            errors.append(f"group {name} names no nodes (node_ids) and no group to rename (rename_from).")
        if color in ROLE_COLORS:
            color = ROLE_COLORS[color]
        if color and not re.fullmatch(r"#[0-9a-fA-F]{3,8}", color):
            errors.append(f"group {name}: color {color!r} is not a hex colour or one of "
                          f"{', '.join(ROLE_COLORS)}.")
        if title.lower() in seen:
            errors.append(f"title {title!r} is used twice in one call.")
        seen.add(title.lower())
        spec = {"title": title[:80], "node_ids": ids}
        if color:
            spec["color"] = color
        if rename:
            spec["rename_from"] = rename
        specs.append(spec)
    return specs, errors
