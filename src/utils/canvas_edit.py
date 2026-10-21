"""canvas_edit — structural edits to the graph open in the user's canvas.

``set_canvas_node_params`` writes values and ``delete_canvas_nodes`` removes
nodes; neither can add a node or change a wire, so "add a hires-fix pass",
"preview instead of saving" or "upscale before saving" could only be answered
by telling the user to do it by hand. This module is the missing third edit:
a list of operations — add a node, connect an output to an input, cut an
input's wire — checked against ComfyUI's ``/object_info`` and applied to the
API-format canvas graph the turn holds.

Pure on purpose: no I/O, no pipeline. The pipeline applies the result to its
copy of the canvas (so ``get_canvas_node`` reads what the edit made) and pushes
one ``{"op": "edit_graph"}`` canvas patch, which the panel replays on the live
graph — and which a headless client (the benchmark harness) can replay on its
own copy the same way.

Operations, applied in order, all or nothing::

    {"op": "add", "class_type": "LatentUpscaleBy", "ref": "up",
     "params": {"scale_by": 1.5}, "near": "5"}
    {"op": "connect", "from": "5", "output": 0, "to": "up", "input": "samples"}
    {"op": "disconnect", "node": "6", "input": "samples"}

``ref`` names a node added in the same call so later operations can wire it;
``output`` is a slot index or the output's name or type.
"""

from __future__ import annotations

import copy

from agenty_core.tools.assembly_deterministic import (
    describe_input_names as _describe_names,
    expand_input_specs as _expand_specs,
)

_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def _specs(schema: dict) -> dict:
    inp = (schema or {}).get("input") or {}
    out = {}
    out.update(inp.get("optional") or {})
    out.update(inp.get("required") or {})
    return out


def _spec_type(spec):
    return spec[0] if isinstance(spec, (list, tuple)) and spec else None


def _is_dynamic(spec) -> bool:
    t = _spec_type(spec)
    return isinstance(t, str) and t.startswith("COMFY_") and t.endswith("_V3")


def _types(t) -> set:
    if isinstance(t, list):
        return {"COMBO"}
    return {p.strip() for p in str(t or "").split(",") if p.strip()}


# The execution wire of a hook pipeline. It is order, not data, so it connects
# to itself and to nothing else - not even to a wildcard, which takes any DATA.
EXEC_TYPE = "AGENTY_EXEC"


def _compatible(out_t, in_t) -> bool:
    a, b = _types(out_t), _types(in_t)
    if EXEC_TYPE in a or EXEC_TYPE in b:
        return EXEC_TYPE in a and EXEC_TYPE in b
    return "*" in a or "*" in b or bool(a & b) or ("COMBO" in b and a & {"COMBO", "STRING"})


def _combo_options(spec):
    t = _spec_type(spec)
    if isinstance(t, list):
        return t
    if t == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options") or []
    return None


def _default(spec):
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return opts["default"]
    combo = _combo_options(spec)
    return combo[0] if combo else None


def _schema_inputs(schema: dict) -> tuple[dict, dict]:
    inp = (schema or {}).get("input") or {}
    return inp.get("required") or {}, inp.get("optional") or {}


def _input_spec(specs: dict, name: str, values: dict | None = None):
    """The spec for *name*, or None when the node has no such input.

    Beside the node's own inputs this knows the dotted ones: a V3 autogrow slot
    (``anchors.anchor0``), a member of a dynamic combo's selected option
    (``model.duration``; the option is ``values[<combo>]``, else the combo's
    default) and a slot of an autogrow group inside such an option
    (``model.reference_images.image_1``). A name that belongs to an option that is
    not selected, or to no option, is None - it would reach ComfyUI as a key the
    node never binds."""
    if name in specs:
        return specs[name]
    if "." not in name:
        return None
    expanded = _expand_specs(specs, None, values)
    if name in expanded:
        return expanded[name]
    head = name.split(".", 1)[0]
    # A growing input whose slots the schema does not list (a prefix template
    # with no maximum): its slots are named at run time, so any is allowed.
    if head in specs and _is_dynamic(specs[head]) and _spec_type(specs[head]) != "COMFY_DYNAMICCOMBO_V3":
        return _autogrow_member(specs[head]) or ["*", {}]
    return None


def _valid_names(specs: dict, values: dict | None = None) -> str:
    """The input names a node takes, as the prompt spells them, for an error."""
    return ", ".join(_describe_names(_expand_specs(specs, None, values))) or "none"


def _unknown_input_error(where: str, cls: str, name: str, specs: dict, values: dict | None) -> str:
    head, _, rest = name.partition(".")
    hint = ""
    spec = specs.get(head)
    if rest and _spec_type(spec) == "COMFY_DYNAMICCOMBO_V3":
        chosen = (values or {}).get(head)
        others = [str(o.get("key")) for o in ((spec[1] or {}).get("options") or [])
                  if isinstance(o, dict) and rest in _option_names(o)]
        if others and chosen not in others:
            hint = (f" '{name}' belongs to option {others} of '{head}', but '{head}' is "
                    f"{chosen!r}; set '{head}' first.")
        elif not isinstance(chosen, str):
            hint = f" '{head}' needs one of its options as a value first."
    elif "." not in name and any(("." in k and k.endswith("." + name)) for k in
                                 _expand_specs(specs, None, values)):
        full = next(k for k in _expand_specs(specs, None, values) if k.endswith("." + name))
        hint = f" Dotted address: '{full}'."
    return (f"{where}: {cls} has no input '{name}'.{hint} "
            f"It takes: {_valid_names(specs, values)}.")


def _option_names(opt: dict) -> set:
    """The input names one option of a dynamic combo brings, nested slots as dotted
    suffixes (``reference_images.image_1``)."""
    inner = opt.get("inputs") or {}
    return set(_expand_specs(inner.get("required"), inner.get("optional"), {}))


def _autogrow_member(spec):
    """The spec of one slot of a growing input, from its template - or None."""
    try:
        template = (spec[1] or {}).get("template") or {}
        inputs = template.get("input") or {}
        members = {**(inputs.get("optional") or {}), **(inputs.get("required") or {})}
        if len(members) == 1:
            return next(iter(members.values()))
    except Exception:  # noqa: BLE001 - an unreadable template is just an untyped slot
        return None
    return None


def _member_spec(specs: dict, name: str, values: dict):
    """The spec of a dynamic combo's member - ``model.duration`` - for the option
    in force (``values[head]``, else the default), or None; also reaches a slot of
    an autogrow group nested in the option.

    `_input_spec` only knows such a name is allowed; without its own spec a
    value written there was never checked, and a Wan 3.0 ``model.duration`` of
    ``5`` went onto the canvas where the menu holds ``"5"`` - which ComfyUI
    refuses ("Value not in list") when the graph is queued."""
    if "." not in name:
        return None
    return _expand_specs(specs, None, values).get(name)


def _dynamic_defaults(name: str, spec, key=None) -> dict:
    """A required dynamic combo's value and its option's member defaults —
    ``{"model": "wan3.0-video", "model.resolution": "1080P", …}``.

    Left out, an added Wan 3.0 node had no `model` at all, so nothing on it
    showed a prompt, a resolution or a duration: the agent read the node as
    having none ("animates purely from the first frame") and stored that as a
    fact. And ComfyUI refuses the graph without the value."""
    options = [o for o in ((spec[1] or {}).get("options") or []) if isinstance(o, dict)] \
        if len(spec) > 1 and isinstance(spec[1], dict) else []
    opt = next((o for o in options if o.get("key") == key), None) or (options[0] if options else None)
    if opt is None:
        return {}
    out = {name: opt.get("key")}
    for member, mspec in ((opt.get("inputs") or {}).get("required") or {}).items():
        mt = _spec_type(mspec)
        if isinstance(mt, list) or mt in _WIDGET_TYPES:
            d = _default(mspec)
            if d is not None:
                out[f"{name}.{member}"] = d
    return out


def coerce_value(spec, value):
    """``(value, error)``: *value* as the input takes it, or the reason it can't.

    The usual slip is the type, not the value: a number written into a menu of
    strings ("5" for 5) or the other way round, which ComfyUI rejects although
    the option is there. Those are matched by their text. Number inputs take a
    numeric string as the number."""
    combo = _combo_options(spec)
    if combo is not None:
        if value in combo:
            return value, None
        same = next((o for o in combo if str(o) == str(value)), None)
        if same is not None:
            return same, None
        near = [o for o in combo if str(value).lower() in str(o).lower()][:5]
        return value, (f"= {value!r} is not an option"
                       + (f"; close: {near}" if near else f"; e.g. {combo[:5]}"))
    t = _spec_type(spec)
    if isinstance(value, str) and not isinstance(value, bool) and t in ("INT", "FLOAT"):
        try:
            num = float(value.strip())
            value = int(num) if t == "INT" and num.is_integer() else num
        except ValueError:
            return value, f"= {value!r} is not a number"
    # Out of range is refused at queue time too: ImageBatchMulti's inputcount
    # set to 1 (its minimum is 2) would have stopped every branch it feeds.
    if t in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        lo, hi = opts.get("min"), opts.get("max")
        if lo is not None and value < lo:
            return value, f"= {value!r} is below its minimum of {lo}"
        if hi is not None and value > hi:
            return value, f"= {value!r} is above its maximum of {hi}"
    return value, None


def coerce_params(schema: dict, params: dict, current: dict | None = None) -> tuple[dict, list[str]]:
    """``(params, errors)``: *params* for a node of *schema*, each value as its
    input takes it. Names the schema does not know pass through untouched
    (the frontend has widgets of its own, e.g. ``control_after_generate``)."""
    specs = _specs(schema)
    values = {**(current or {}), **params}
    out, errors = {}, []
    for name, value in params.items():
        spec = specs.get(name) if name in specs else _member_spec(specs, name, values)
        if spec is None or _is_link(value):
            out[name] = value
            continue
        fixed, err = coerce_value(spec, value)
        if err:
            errors.append(f"{name} {err}")
        out[name] = fixed
    return out, errors


def _next_id(graph: dict) -> int:
    nums = [int(k) for k in graph if str(k).isdigit()]
    return (max(nums) if nums else 0) + 1


def unwired_required(graph: dict, node_id: str, object_info: dict) -> list[str]:
    """Required LINK inputs of *node_id* that nothing feeds — why it won't run."""
    node = graph.get(str(node_id)) or {}
    schema = object_info.get(node.get("class_type")) or {}
    req = ((schema.get("input") or {}).get("required") or {})
    inputs = node.get("inputs") or {}
    out = []
    for name, spec in req.items():
        t = _spec_type(spec)
        is_value = isinstance(t, list) or t in _WIDGET_TYPES or _is_dynamic(spec)
        if not is_value and name not in inputs:
            out.append(f"{name} ({t})")
    return out


def wire_candidates(graph: dict, node_id: str, object_info: dict, limit: int = 3) -> dict:
    """For each unfed required input of *node_id* (as :func:`unwired_required`
    names it): the outputs already in the graph that carry its type -
    ``{"vae (VAE)": ["node 4 (CheckpointLoaderSimple) output 2"]}``. Empty where
    nothing in the graph makes that type: a node has to be added first."""
    out: dict = {}
    for name in unwired_required(graph, node_id, object_info):
        want = name[name.rfind("(") + 1:-1] if name.endswith(")") and "(" in name else ""
        found = []
        for other_id, other in (graph or {}).items():
            if str(other_id) == str(node_id) or not isinstance(other, dict):
                continue
            outputs = (object_info.get(other.get("class_type")) or {}).get("output") or []
            for slot, kind in enumerate(outputs):
                if want and str(kind) == want:
                    found.append(f"node {other_id} ({other.get('class_type')}) output {slot}")
        if found:
            out[name] = found[:limit]
    return out


def plan(graph: dict, ops: list, object_info: dict) -> dict:
    """Check and apply *ops* to a copy of *graph*.

    Returns ``{"ok": bool, "errors": [...], "graph": new_graph, "ops": [...],
    "added": {ref: node_id}, "touched": [node ids]}``. On any error nothing is
    applied (``graph`` is the input, unchanged) — a half-made edit is worse than
    none, because the next call cannot tell which half happened.
    """
    g = copy.deepcopy(graph if isinstance(graph, dict) else {})
    errors: list[str] = []
    resolved: list[dict] = []
    added: dict[str, str] = {}
    touched: list[str] = []
    removed: set[str] = set()

    def node_of(ref) -> str | None:
        r = str(ref).strip().lstrip("#")
        if r in added:
            return added[r]
        return r if r in g else None

    if not isinstance(ops, list) or not ops:
        return {"ok": False, "errors": ["ops must be a non-empty list of operations."],
                "graph": graph, "ops": [], "added": {}, "touched": []}

    for i, op in enumerate(ops, 1):
        where = f"op {i}"
        if not isinstance(op, dict):
            errors.append(f"{where}: each operation is an object, got {type(op).__name__}.")
            continue
        kind = str(op.get("op") or "").lower()

        if kind == "add":
            cls = str(op.get("class_type") or "")
            schema = object_info.get(cls)
            if not schema:
                errors.append(f"{where}: node type '{cls}' is not installed in this ComfyUI.")
                continue
            specs = _specs(schema)
            params = op.get("params") or {}
            if not isinstance(params, dict):
                errors.append(f"{where}: params must be an object of widget -> value.")
                continue
            inputs = {}
            for name, spec in ((schema.get("input") or {}).get("required") or {}).items():
                t = _spec_type(spec)
                if isinstance(t, list) or t in _WIDGET_TYPES:
                    d = _default(spec)
                    if d is not None:
                        inputs[name] = d
                elif t == "COMFY_DYNAMICCOMBO_V3":
                    inputs.update(_dynamic_defaults(name, spec, params.get(name)))
            bad = False
            # A dynamic combo's value first: its option decides which dotted
            # names exist, on the page as well as here.
            params = dict(sorted(params.items(), key=lambda kv: str(kv[0]).count(".")))
            in_force = {**inputs, **params}
            for name in params:
                if _input_spec(specs, name, in_force) is None:
                    errors.append(_unknown_input_error(where, cls, name, specs, in_force))
                    bad = True
            params, wrong = coerce_params(schema, params, inputs)
            for w in wrong:
                errors.append(f"{where}: {cls}.{w}.")
                bad = True
            if bad:
                continue
            inputs.update(params)
            defaults = {k: v for k, v in inputs.items() if k not in params}
            # "id": the id of a node removed earlier in this call - a replacement
            # keeps it, so hooks, notes and the user's "#7" still name that place.
            want_id = str(op.get("id") or "").strip()
            if want_id and (want_id in g or not want_id.isdigit()):
                errors.append(f"{where}: id {want_id} is taken (or not a number); "
                              f"remove that node first or leave id out.")
                continue
            nid = want_id or str(max(_next_id(g), _next_id({k: 1 for k in removed})))
            g[nid] = {"class_type": cls, "inputs": inputs}
            ref = str(op.get("ref") or nid)
            added[ref] = nid
            added[nid] = nid
            near = node_of(op.get("near")) if op.get("near") is not None else None
            entry = {"op": "add", "node_id": nid, "class_type": cls,
                     "params": params, "defaults": defaults, "near": near}
            if op.get("at") is not None and str(op.get("at")) in removed:
                entry["at"] = str(op.get("at"))      # takes that node's place on the canvas
            resolved.append(entry)
            touched.append(nid)

        elif kind == "connect":
            src, dst = node_of(op.get("from")), node_of(op.get("to"))
            name = str(op.get("input") or "")
            if src is None or dst is None:
                missing = [str(op.get(k)) for k, v in (("from", src), ("to", dst)) if v is None]
                errors.append(f"{where}: no node {', '.join(missing)} on the canvas (or added earlier in this call).")
                continue
            s_schema = object_info.get(g[src].get("class_type")) or {}
            d_schema = object_info.get(g[dst].get("class_type")) or {}
            outs = list(s_schema.get("output") or [])
            names = list(s_schema.get("output_name") or [])
            want = op.get("output", 0)
            slot = None
            if isinstance(want, int) or (isinstance(want, str) and want.isdigit()):
                slot = int(want)
            else:
                for j, (t, n) in enumerate(zip(outs, names or outs)):
                    if str(want) in (str(t), str(n)):
                        slot = j
                        break
            if slot is None or not (0 <= slot < len(outs)):
                listing = ", ".join(f"{j}={n}({t})" for j, (t, n) in enumerate(zip(outs, names or outs)))
                errors.append(f"{where}: #{src} {g[src]['class_type']} has no output {want!r} "
                              f"(outputs: {listing or 'none'}).")
                continue
            d_specs = _specs(d_schema)
            d_values = g[dst].get("inputs") or {}
            spec = _input_spec(d_specs, name, d_values)
            if spec is None:
                errors.append(_unknown_input_error(f"{where}: #{dst}", g[dst]["class_type"],
                                                   name, d_specs, d_values))
                continue
            if not _compatible(outs[slot], _spec_type(spec)):
                errors.append(f"{where}: #{src} output {slot} is {outs[slot]}, but "
                              f"#{dst}.{name} takes {_spec_type(spec)}.")
                continue
            g[dst].setdefault("inputs", {})[name] = [src, slot]
            resolved.append({"op": "connect", "from": src, "output": slot, "to": dst, "input": name,
                             "slot_type": outs[slot] if isinstance(outs[slot], str) else "*"})
            touched += [src, dst]

        elif kind == "disconnect":
            nid = node_of(op.get("node"))
            name = str(op.get("input") or "")
            if nid is None:
                errors.append(f"{where}: no node {op.get('node')} on the canvas.")
                continue
            if not _is_link((g[nid].get("inputs") or {}).get(name)):
                errors.append(f"{where}: #{nid}.{name} is not wired, so there is nothing to disconnect.")
                continue
            del g[nid]["inputs"][name]
            resolved.append({"op": "disconnect", "node": nid, "input": name})
            touched.append(nid)

        elif kind == "remove":
            # Part of a replacement (replacement_ops). Deleting on its own is
            # delete_canvas_nodes, which says what each deleted node took with it.
            nid = node_of(op.get("node"))
            if nid is None:
                errors.append(f"{where}: no node {op.get('node')} on the canvas.")
                continue
            del g[nid]
            for other in g.values():
                ins = other.get("inputs") or {}
                for k in [k for k, v in ins.items() if _is_link(v) and str(v[0]) == nid]:
                    del ins[k]
            removed.add(nid)
            resolved.append({"op": "remove", "node": nid})

        else:
            errors.append(f"{where}: unknown op {op.get('op')!r} — use add, connect or disconnect "
                          f"(delete_canvas_nodes removes nodes, set_canvas_node_params sets values).")

    if errors:
        return {"ok": False, "errors": errors, "graph": graph, "ops": [], "added": {}, "touched": []}
    refs = {k: v for k, v in added.items() if k != v}
    return {"ok": True, "errors": [], "graph": g, "ops": resolved, "added": refs,
            "touched": [n for n in dict.fromkeys(touched) if n in g]}


def _is_widget_spec(spec) -> bool:
    t = _spec_type(spec)
    return isinstance(t, list) or t in _WIDGET_TYPES or _is_dynamic(spec)


def replacement_ops(graph: dict, node_id: str, class_type: str, object_info: dict,
                    params: dict | None = None, wires: dict | None = None) -> dict:
    """The ops that put a node of *class_type* in the place of *node_id*: the
    same id and spot on the canvas, its wires carried over, its values kept
    where the new node has an input of that name that takes them.

    A swap was add + rewire + delete by hand, three calls in which the model had
    to read both nodes' inputs and every consumer's - and it did not: asked to put
    a Seedream node where a GPT Image node was, one built a separate workflow and
    ran it beside the graph (2026-10-14).

    A wire INTO the old node goes to the new input of the same name, else to the
    first free input that takes its type (links before widgets, required before
    optional). A wire OUT leaves the new output at the same slot if that carries
    the type, else the first that does. *wires* overrides by the old input's
    name: ``{"images": "image"}``, or ``""`` to leave that wire behind. *params*
    sets values on the new node and wins over a carried one.

    Returns ``{"ok", "errors", "ops", "carried", "dropped_values", "wired_in",
    "wired_out", "left_behind"}``. Not ok when a wire cannot be carried and is not
    named in *wires*: a replacement that silently loses a feed is not one.
    """
    nid = str(node_id).strip().lstrip("#")
    g = graph if isinstance(graph, dict) else {}
    out = {"ok": False, "errors": [], "ops": [], "carried": {}, "dropped_values": [],
           "wired_in": [], "wired_out": [], "left_behind": []}
    old = g.get(nid)
    if not isinstance(old, dict):
        out["errors"].append(f"no node {nid} on the canvas.")
        return out
    new_schema = object_info.get(class_type)
    if not new_schema:
        out["errors"].append(f"node type '{class_type}' is not installed in this ComfyUI.")
        return out
    old_schema = object_info.get(old.get("class_type")) or {}
    new_specs = _specs(new_schema)
    listing = ", ".join(f"{n} ({_spec_type(s) if not isinstance(_spec_type(s), list) else 'COMBO'})"
                        for n, s in sorted(new_specs.items()))
    required = set(((new_schema.get("input") or {}).get("required") or {}))
    old_inputs = old.get("inputs") or {}
    wires = {str(k): ("" if v is None else str(v)) for k, v in (wires or {}).items()}
    params = dict(params or {})
    unknown = [k for k in params if _input_spec(new_specs, k, params) is None]
    if unknown:
        out["errors"].append(f"{class_type} has no input {', '.join(map(repr, unknown))} "
                             f"(it takes: {_valid_names(new_specs, params)}).")
    for k, target in wires.items():
        if not _is_link(old_inputs.get(k)):
            out["errors"].append(f"wires names '{k}', but nothing is wired into #{nid}.{k}.")
        elif target and _input_spec(new_specs, target, params) is None:
            out["errors"].append(f"wires sends '{k}' to '{target}', which {class_type} does not have "
                                 f"(it has: {listing or 'none'}).")

    def out_type(src_id, slot):
        src = g.get(str(src_id)) or {}
        outs = (object_info.get(src.get("class_type")) or {}).get("output") or []
        return outs[slot] if 0 <= slot < len(outs) else None

    def same_kind(t, spec) -> bool:
        # A wire may go into a widget (a STRING into a prompt) only when it IS
        # that kind - never because the widget's side says "*".
        return bool((_types(t) - {"*"}) & _types(_spec_type(spec)))

    # Wires in.
    taken: set = set()
    connects: list = []
    # What the new node's dotted inputs are judged against: the values it will
    # hold, so a wire into ``model.reference_images.image_1`` finds that slot on
    # the same dynamic combo option.
    carried_values = {**{k: v for k, v in old_inputs.items() if not _is_link(v)}, **params}
    order = sorted(new_specs, key=lambda n: (_is_widget_spec(new_specs[n]), n not in required))
    for name, value in old_inputs.items():
        if not _is_link(value):
            continue
        src, slot = str(value[0]), int(value[1])
        t = out_type(src, slot)
        if name in wires:
            target = wires[name] or None
            if target is None:
                out["left_behind"].append(f"#{src} → {name}")
                continue
        elif (_input_spec(new_specs, name, carried_values) is not None
              and _compatible(t, _spec_type(_input_spec(new_specs, name, carried_values)))):
            target = name
        else:
            target = next((n for n in order if n not in taken and n not in params
                           and n not in old_inputs
                           and _compatible(t, _spec_type(new_specs[n]))
                           and (not _is_widget_spec(new_specs[n]) or same_kind(t, new_specs[n]))),
                          None)
        if target is None or target in taken:
            out["errors"].append(
                f"the wire from #{src} ({t}) into '{name}' has nowhere to go on {class_type} "
                f"(its inputs: {listing or 'none'}). Name the input it should feed in wires, "
                f"e.g. {{\"{name}\": \"<input>\"}}, or \"\" to leave that wire behind.")
            continue
        taken.add(target)
        connects.append({"op": "connect", "from": src, "output": slot, "to": "new", "input": target})
        out["wired_in"].append(f"#{src} → {target}" + (f" (was {name})" if target != name else ""))

    # Values, where the new node has an input of that name that takes them.
    for name, value in old_inputs.items():
        if _is_link(value) or name in params or name in taken:
            continue
        spec = new_specs.get(name)
        if spec is None or not _is_widget_spec(spec) or _is_dynamic(spec):
            out["dropped_values"].append(name)
            continue
        fixed, err = coerce_value(spec, value)
        if err:
            out["dropped_values"].append(f"{name} {err}")
        else:
            out["carried"][name] = fixed

    # Wires out.
    new_outs = list(new_schema.get("output") or [])
    old_outs = list(old_schema.get("output") or [])
    for cid, cnode in g.items():
        if str(cid) == nid or not isinstance(cnode, dict):
            continue
        for iname, v in (cnode.get("inputs") or {}).items():
            if not (_is_link(v) and str(v[0]) == nid):
                continue
            slot = int(v[1])
            cspec = _input_spec(_specs(object_info.get(cnode.get("class_type")) or {}), iname,
                               cnode.get("inputs"))
            want = (_spec_type(cspec) if cspec is not None
                    else (old_outs[slot] if slot < len(old_outs) else None))
            if (want is None or _types(want) == {"*"}) and slot < len(old_outs):
                want = old_outs[slot]        # into a wildcard: keep the type that flowed
            pick = (slot if slot < len(new_outs) and _compatible(new_outs[slot], want)
                    else next((j for j, t in enumerate(new_outs) if _compatible(t, want)), None))
            if pick is None:
                out["errors"].append(
                    f"#{cid} {cnode.get('class_type')}.{iname} takes {want} from #{nid}, and "
                    f"{class_type} has no output of that type "
                    f"(outputs: {', '.join(map(str, new_outs)) or 'none'}).")
                continue
            connects.append({"op": "connect", "from": "new", "output": pick, "to": str(cid), "input": iname})
            out["wired_out"].append(f"#{cid}.{iname}")

    if out["errors"]:
        return out
    out["ops"] = [{"op": "remove", "node": nid},
                  {"op": "add", "class_type": class_type, "ref": "new", "id": nid, "at": nid,
                   "params": {**out["carried"], **params}}] + connects
    out["ok"] = True
    return out


def block_layout(workflow: dict, added: dict, prefix: str = "wf") -> dict:
    """How an inserted workflow should be laid out on the canvas, or ``{}``.

    The same arrangement the agent's own workflows open with (agenty_core's
    graph_groups: a column per step, a row per node), as slots - band, column,
    row - for each node, keyed by the id the node has ON THE CANVAS. The panel
    turns slots into positions, because only the canvas knows how big its nodes
    really are; it also puts the block below what is already there and draws a
    group around it.

    *added* is :func:`plan`'s ``added``: the ref each node was added under
    (``wf<file id>``) to its canvas id.
    """
    try:
        from agenty_core.tools.comfyui import _api_to_graph  # noqa: PLC0415
        hint = (_api_to_graph(workflow).get("extra") or {}).get("agentY_layout") or {}
    except Exception:  # noqa: BLE001 - without a layout the nodes still land, just plainly
        return {}
    slots = {}
    for file_id, slot in (hint.get("slots") or {}).items():
        canvas_id = (added or {}).get(f"{prefix}{file_id}")
        if canvas_id is not None and isinstance(slot, (list, tuple)) and len(slot) >= 3:
            slots[str(canvas_id)] = [int(slot[0]), int(slot[1]), int(slot[2])]
    if not slots:
        return {}
    return {"slots": slots, "gaps": dict(hint.get("gaps") or {}), "prefix": "agent"}


def subgraph(graph: dict, node_ids) -> dict:
    """The nodes *node_ids* of an API-format graph and everything upstream of
    them - what has to run for those nodes to run. Ids not in the graph (deleted,
    bypassed or muted since) are left out; an empty dict when none remain."""
    graph = graph if isinstance(graph, dict) else {}
    keep: dict = {}
    todo = [str(n) for n in (node_ids or []) if str(n) in graph]
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep[nid] = copy.deepcopy(graph[nid])
        for value in (graph[nid].get("inputs") or {}).values():
            if _is_link(value) and str(value[0]) in graph and str(value[0]) not in keep:
                todo.append(str(value[0]))
    return keep


def ops_for_workflow(workflow: dict, prefix: str = "wf") -> list:
    """A built workflow (API format) as the ops that put it on a canvas, wired.

    One ``add`` per node - its values as ``params`` - and one ``connect`` per
    link, in an order :func:`plan` can apply: every node before the wires, and
    each node after the one it is placed beside. Nothing is decided here that
    the workflow does not already say: the model is not asked to restate a
    workflow it has just built, node by node and wire by wire.

    Raises ``ValueError`` for anything that is not an API-format workflow.
    """
    if not isinstance(workflow, dict) or isinstance(workflow.get("nodes"), list):
        raise ValueError("not an API-format workflow (a saved canvas graph cannot be inserted this way)")
    nodes = {str(k): v for k, v in workflow.items()
             if isinstance(v, dict) and v.get("class_type")}
    if not nodes:
        raise ValueError("the workflow has no nodes")

    def sources(nid: str) -> list:
        return [str(v[0]) for v in (nodes[nid].get("inputs") or {}).values()
                if _is_link(v) and str(v[0]) in nodes]

    # Upstream first, so a node can be placed beside the one that feeds it.
    order, seen = [], set()

    def visit(nid: str, trail: tuple = ()) -> None:
        if nid in seen or nid in trail:
            return
        for src in sources(nid):
            visit(src, trail + (nid,))
        seen.add(nid)
        order.append(nid)

    def key(n: str):
        return (0, int(n)) if n.isdigit() else (1, n)

    for nid in sorted(nodes, key=key):
        visit(nid)

    ref = {nid: f"{prefix}{nid}" for nid in nodes}
    ops: list = []
    for nid in order:
        node = nodes[nid]
        params = {k: v for k, v in (node.get("inputs") or {}).items() if not _is_link(v)}
        op = {"op": "add", "class_type": node["class_type"], "ref": ref[nid], "params": params}
        feeds = sources(nid)
        if feeds:
            op["near"] = ref[feeds[0]]
        ops.append(op)
    for nid in order:
        for name, value in (nodes[nid].get("inputs") or {}).items():
            if _is_link(value) and str(value[0]) in nodes:
                ops.append({"op": "connect", "from": ref[str(value[0])], "output": int(value[1]),
                            "to": ref[nid], "input": name})
    return ops


def apply_patch(graph: dict, patch: dict) -> dict:
    """Replay a pushed canvas patch on an API-format graph (in place; returned).

    What the panel does to the live graph, for a client that only has the API
    form — the benchmark harness, and the pipeline's own copy of the canvas.
    """
    op = patch.get("op")
    if op is None and isinstance(patch.get("params"), dict):
        node = graph.get(str(patch.get("node_id")))
        if node is not None:
            ins = node.setdefault("inputs", {})
            for k, v in patch["params"].items():
                if not _is_link(ins.get(k)):
                    ins[k] = v
    elif op == "delete_nodes":
        gone = {str(n) for n in patch.get("node_ids") or []}
        for n in gone:
            graph.pop(n, None)
        for node in graph.values():
            ins = node.get("inputs") or {}
            for k in [k for k, v in ins.items() if _is_link(v) and str(v[0]) in gone]:
                del ins[k]
    elif op == "edit_graph":
        for o in patch.get("ops") or []:
            if o.get("op") == "add":
                graph[str(o["node_id"])] = {"class_type": o["class_type"],
                                            "inputs": dict(o.get("defaults") or {}, **(o.get("params") or {}))}
            elif o.get("op") == "connect" and str(o.get("to")) in graph:
                graph[str(o["to"])].setdefault("inputs", {})[o["input"]] = [str(o["from"]), int(o["output"])]
            elif o.get("op") == "disconnect" and str(o.get("node")) in graph:
                (graph[str(o["node"])].get("inputs") or {}).pop(o.get("input"), None)
            elif o.get("op") == "remove":
                gone = str(o.get("node"))
                graph.pop(gone, None)
                for node in graph.values():
                    ins = node.get("inputs") or {}
                    for k in [k for k, v in ins.items() if _is_link(v) and str(v[0]) == gone]:
                        del ins[k]
    return graph
