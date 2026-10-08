"""Which node loads and which node saves each kind of file.

By default agentY takes whatever a workflow was built with and, for results it
drops on the canvas, the first loader this ComfyUI has. Settings ▸ Load & save
nodes lets the user pick instead — a different saver for images (bEpic's "Send
to Image Viewer", an EXR writer), a path loader for videos, and so on.

Three things live here:

* :func:`choices` — the candidates per kind, read from ComfyUI's own node list;
* :func:`preferred` — what the settings say (``""`` = automatic);
* :func:`apply_savers` — swap the save nodes of a graph for the chosen ones,
  called where a workflow is built and again by the executor before every
  submission, so no run gets past it.

A swap happens only when it is certainly the same job: the old node is a saver
of that kind, and the chosen node has an input for the very wire the old one
was saving. Anything else is left as it is.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("agentY.media_nodes")

KINDS = ("image", "video", "audio")
_WIRE = {"image": ("IMAGE",), "video": ("VIDEO", "IMAGE"), "audio": ("AUDIO",)}
_BUCKET = {"images": "image", "videos": "video", "audio": "audio"}
_WILD = {"*", "COMFY_MATCHTYPE_V3", "ANY"}
_SCALARS = {"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL", "COMBO", "NUMBER",
            "COMFY_DYNAMICCOMBO_V3", "COMFY_AUTOGROW_V3"}
# The input a saver names its file with, best first.
_NAME_INPUTS = ("filename_prefix", "filename", "prefix", "output_path", "file_name")
# The widget a loader keeps its file in.
FILE_WIDGETS = ("image", "video", "audio", "file", "path", "image_path", "video_path",
                "audio_path", "file_path", "filepath", "filename")

_choices_cache: dict = {}


def _types(schema: dict) -> dict:
    """``{input name: declared type}`` for one node (a combo's list is ``COMBO``)."""
    out = {}
    for section in ("required", "optional"):
        for name, spec in (((schema or {}).get("input") or {}).get(section) or {}).items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            out[name] = "COMBO" if isinstance(t, (list, tuple)) else str(t)
    return out


def _required(schema: dict) -> dict:
    return dict((((schema or {}).get("input") or {}).get("required")) or {})


def _is_wire(type_name: str) -> bool:
    return type_name not in _SCALARS


def _bucket(class_type: str) -> str:
    try:
        from agenty_core.tools.comfyui import _agent_media_bucket
        return _BUCKET.get(_agent_media_bucket(class_type), "")
    except Exception:  # noqa: BLE001
        return "image"


def _label(class_type: str, schema: dict) -> str:
    name = str((schema or {}).get("display_name") or "").strip()
    return f"{name}  ({class_type})" if name and name != class_type else class_type


# ── what can be chosen ──────────────────────────────────────────────────────

def saver_kinds(class_type: str, schema: dict) -> list[str]:
    """The kinds *class_type* can save: an output node that names its file and
    takes that kind's wire (or any wire)."""
    if not (schema or {}).get("output_node"):
        return []
    types = _types(schema)
    if not any(n in types for n in _NAME_INPUTS):
        return []                      # a preview: it shows, it does not save
    wires = {t for t in types.values() if _is_wire(t)}
    if wires & _WILD:
        # Takes anything, e.g. bEpicSendToViewer. Stricter about the name input:
        # debug nodes ("show any") take anything too and save nothing.
        return list(KINDS) if "filename_prefix" in types else []
    mine = _bucket(class_type)
    return [k for k in KINDS if k == mine and wires & set(_WIRE[k])]


def loader_kinds(class_type: str, schema: dict) -> list[str]:
    """The kinds *class_type* can load: needs nothing wired, keeps a file in a
    widget, and hands out that kind's wire."""
    types = _types(schema)
    if any(_is_wire(types.get(n, "")) for n in _required(schema)):
        return []
    if not any(n in types for n in FILE_WIDGETS):
        return []
    text = (class_type + " " + str((schema or {}).get("display_name") or "")).lower()
    if "load" not in text:
        return []
    outs = [str(o) for o in ((schema or {}).get("output") or [])]
    for kind in ("video", "audio", "image"):      # "Load Video (Path)" is not an image loader
        if kind in text and set(outs) & set(_WIRE[kind]):
            return [kind]
    return []


def choices(object_info: dict | None = None) -> dict:
    """``{kind: {"load": [{id, label}], "save": [...]}}`` for this ComfyUI.

    Empty when ComfyUI cannot be asked. Cached for the life of the host: the
    node list is nine seconds to fetch here and changes only with a restart.
    """
    if object_info is None:
        if _choices_cache:
            return _choices_cache
        try:
            from agenty_core.tools.comfyui import _get_object_info
            object_info = _get_object_info()
        except Exception as exc:  # noqa: BLE001
            logger.debug("media nodes: ComfyUI could not be asked (%s)", exc)
            return {}
    out = {k: {"load": [], "save": []} for k in KINDS}
    for cls, schema in sorted((object_info or {}).items()):
        if not isinstance(schema, dict):
            continue
        entry = {"id": cls, "label": _label(cls, schema)}
        for kind in saver_kinds(cls, schema):
            out[kind]["save"].append(entry)
        for kind in loader_kinds(cls, schema):
            out[kind]["load"].append(entry)
    if object_info and any(v["load"] or v["save"] for v in out.values()):
        _choices_cache.clear()
        _choices_cache.update(out)
    return out


# ── what was chosen ─────────────────────────────────────────────────────────

def preferred(kind: str, role: str) -> str:
    """The class chosen for *kind* (``image``…) and *role* (``load`` / ``save``),
    or ``""`` for automatic."""
    try:
        from src.agent import _load_settings
        chosen = ((_load_settings() or {}).get("media_nodes") or {}).get(f"{kind}_{role}")
    except Exception:  # noqa: BLE001
        return ""
    return str(chosen or "").strip()


def loader_candidates(kind: str, defaults: list) -> list:
    """*defaults* with the chosen loader in front."""
    first = preferred(kind, "load")
    return ([first] if first else []) + [c for c in defaults if c != first]


def file_widget(schema: dict) -> tuple[str, bool]:
    """``(widget name, takes an absolute path)`` for a loader; ``("", False)``
    when it has none. A free-text widget reads the file where it is; a list
    names a copy in ComfyUI's input folder."""
    types = _types(schema)
    for name in FILE_WIDGETS:
        if name in types:
            return name, types[name] == "STRING"
    return "", False


# ── swapping the savers of a graph ──────────────────────────────────────────

def _schema(class_type: str) -> dict:
    from src.utils.preflight import _schema as one
    return one(class_type)


def _default_inputs(class_type: str) -> dict:
    try:
        from agenty_core.tools.comfyui import node_default_inputs
        return dict(node_default_inputs(class_type) or {})
    except Exception:  # noqa: BLE001
        return {}


def _swap(node: dict, kind: str, new_class: str, old_schema: dict, new_schema: dict) -> dict | None:
    """*node* rebuilt as *new_class*, or None when that is not the same job."""
    old_types, new_types = _types(old_schema), _types(new_schema)
    inputs = node.get("inputs") or {}
    # The wire being saved: the old node's linked input of this kind's type.
    source = next(((name, old_types.get(name)) for name, value in inputs.items()
                   if isinstance(value, list) and old_types.get(name) in _WIRE[kind]), None)
    if source is None:
        return None
    wire_name, wire_type = source
    target = next((n for n, t in new_types.items() if t == wire_type), None) \
        or next((n for n, t in new_types.items() if t in _WILD), None)
    if target is None:
        return None                    # the chosen node cannot take this wire
    new_inputs = _default_inputs(new_class)
    new_inputs[target] = list(inputs[wire_name])
    # Same file name, under whatever the new node calls it.
    old_name = next((inputs[n] for n in _NAME_INPUTS if isinstance(inputs.get(n), (str, list))), None)
    new_name = next((n for n in _NAME_INPUTS if n in new_types), None)
    if old_name is not None and new_name:
        new_inputs[new_name] = old_name
    # Other wires both nodes know by the same name and type (audio beside images).
    for name, value in inputs.items():
        if name != wire_name and isinstance(value, list) and name in new_types \
                and new_types[name] == old_types.get(name):
            new_inputs[name] = list(value)
    # A node that only shows unless told to save must be told to save.
    if new_types.get("save_to_output") in ("BOOLEAN", "BOOL"):
        new_inputs["save_to_output"] = True
    if kind == "video" and wire_type == "IMAGE" and new_types.get("file_format") == "COMBO":
        options = (_required(new_schema).get("file_format") or [[]])[0]
        if isinstance(options, (list, tuple)) and "mp4" in options:
            new_inputs["file_format"] = "mp4"
    swapped = {**node, "class_type": new_class, "inputs": new_inputs}
    meta = dict(swapped.get("_meta") or {})
    meta.pop("title", None)            # the old node's name would now be a lie
    if meta:
        swapped["_meta"] = meta
    else:
        swapped.pop("_meta", None)
    return swapped


def apply_savers(graph: dict, schema_of=None) -> dict:
    """Swap *graph*'s save nodes for the chosen ones, in place.

    Returns ``{node id: (old class, new class)}`` for what changed. A node keeps
    its id, so everything that refers to it - the briefing, a canvas mapping,
    the output-name rule - still does.
    """
    changed = {}
    wanted = {k: preferred(k, "save") for k in KINDS}
    if not any(wanted.values()) or not isinstance(graph, dict):
        return changed
    schema_of = schema_of or _schema
    for nid, node in list(graph.items()):
        if not isinstance(node, dict):
            continue
        old = str(node.get("class_type") or "")
        try:
            old_schema = schema_of(old)
            if not old_schema or set(_types(old_schema).values()) & _WILD and old_schema.get("output_node"):
                continue               # unknown, or already a take-anything saver
            kinds = saver_kinds(old, old_schema)
            if len(kinds) != 1:
                continue
            kind = kinds[0]
            new = wanted.get(kind) or ""
            if not new or new == old:
                continue
            new_schema = schema_of(new)
            if not new_schema or kind not in saver_kinds(new, new_schema):
                continue               # not installed here, or not a saver of this kind
            swapped = _swap(node, kind, new, old_schema, new_schema)
        except Exception as exc:  # noqa: BLE001 - a preference must never cost a run
            logger.warning("media nodes: left node %s (%s) as it is: %s", nid, old, exc)
            continue
        if swapped is not None:
            graph[nid] = swapped
            changed[str(nid)] = (old, new)
    return changed
