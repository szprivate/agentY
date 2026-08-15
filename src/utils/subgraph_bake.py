"""Bake a chain of generated stage-workflows into a ComfyUI *subgraph* canvas.

When a make_workflow hook has ``bake`` on, the orchestrator generates
a workflow per stage and then asks to "bake" the chain: each stage's workflow is
nested into a ComfyUI **subgraph** whose exposed inputs/outputs match the hook's
slots, the subgraph instances are placed on one canvas and wired to mirror the
hook chain. The result is a self-contained native workflow the user can re-run
without the agent.

This module is the pure builder — no network, no agent. It consumes *graph*-format
stage workflows (nodes + links, v0.4 — the tool converts API format first via
``agenty_core``'s ``_api_to_graph``) and produces a single graph-format workflow
with a ``definitions.subgraphs`` block and one subgraph-instance node per stage,
chained together. The shape matches ComfyUI's own serialized subgraphs (see
``blueprints/*.json`` / official ``*_subgraphed.json`` templates), so it loads via
``loadGraphData`` / ``/agent/load_workflow``.

Slots are type-agnostic where a type isn't given, so a stage can export images,
video, and scalars (STRING/INT/FLOAT) alike.
"""
from __future__ import annotations

import copy
import re
import uuid
from typing import Any


# Fixed pseudo-node ids for a subgraph's input/output boundary (ComfyUI convention).
_INPUT_NODE_ID = -10
_OUTPUT_NODE_ID = -20

# A subgraph's name is rendered on the collapsed instance node, in the breadcrumb
# when you open it, and in the node-library entry. All three are small. A stage
# named with the whole directive — "generate a cinematic wide shot of a lighthouse
# at dusk, moody volumetric light, 35mm, …" — makes an unreadable canvas out of
# what baking was supposed to make readable.
_MAX_NAME_WORDS = 5
_MAX_NAME_CHARS = 42

# Openers that carry no information about what the stage DOES. Dropped from the
# front only, so "Generate the reference frames" keeps its verb while "Please
# generate …" and "Then, generate …" lose theirs.
_FILLER = ("please", "now", "then", "next", "first", "also", "just", "kindly",
           "i want you to", "you should", "your task is to", "the task is to",
           "this hook", "this stage", "for this stage", "in this step")

# A name may not END on one of these: cutting at five words lands on them often
# ("Upscale the image 2x and"), and a dangling conjunction reads as truncation
# rather than as a name.
_DANGLING = {"and", "or", "but", "with", "of", "to", "into", "for", "from", "at",
             "in", "on", "by", "as", "the", "a", "an", "then", "so", "that",
             "plus", "using", "via", "per", "its", "their", "this", "these",
             "those", "it", "them", "one"}


def _uuid() -> str:
    return str(uuid.uuid4())


def short_name(text: str, fallback: str = "Baked stage") -> str:
    """A 2–5 word name for a baked stage, from whatever the caller passed.

    The agent is asked for a short functional name and usually gives one. This is
    what happens when it doesn't: the name is derived rather than trusted, because
    a directive pasted in whole is the failure that shows up on the canvas, and it
    shows up on EVERY stage at once.

    Takes the first clause, drops leading filler, and clamps to five words — a
    sentence's first clause is nearly always the thing it asks for.
    """
    raw = " ".join(str(text or "").split())
    if not raw:
        return fallback
    # Filler first, and before the clause split: "Then, animate the scene" splits
    # on that comma into "Then", which names nothing at all.
    changed = True
    while changed:
        changed = False
        low = raw.lower()
        for f in _FILLER:
            if low.startswith(f + " ") or low.startswith(f + ","):
                raw = raw[len(f):].lstrip(" ,").strip()
                changed = True
                break
    # First clause: the ask, before the styling notes and the qualifications.
    clause = re.split(r"[.;:\n]|\s[—–-]\s|,\s", raw, maxsplit=1)[0].strip() or raw
    words = clause.split()[:_MAX_NAME_WORDS]
    # Never end on a dangling word, and never cut mid-word to fit.
    def _trim(ws: list) -> list:
        while ws:
            last = ws[-1].strip(" ,;:-—–").lower()
            # A bare number at the end is the same artefact: "a 5" is where
            # "a 5 second clip" was cut, not a name.
            if last in _DANGLING or last.isdigit():
                ws = ws[:-1]
                continue
            return ws
        return ws
    words = _trim(words)
    while words and len(" ".join(words)) > _MAX_NAME_CHARS and len(words) > 1:
        words = _trim(words[:-1])
    name = " ".join(words).strip(" ,;:-—–")[:_MAX_NAME_CHARS].strip(" ,;:-—–")
    if not name:
        return fallback
    return name[0].upper() + name[1:]


def _as_object_link(link: Any) -> dict | None:
    """Normalise a v0.4 array link ``[id,o,os,t,ts,type]`` (or object link) to the
    object form ComfyUI uses inside a subgraph definition."""
    if isinstance(link, dict):
        return {
            "id": link.get("id"),
            "origin_id": link.get("origin_id"),
            "origin_slot": link.get("origin_slot", 0),
            "target_id": link.get("target_id"),
            "target_slot": link.get("target_slot", 0),
            "type": link.get("type", "*"),
        }
    if isinstance(link, list) and len(link) >= 6:
        return {"id": link[0], "origin_id": link[1], "origin_slot": link[2],
                "target_id": link[3], "target_slot": link[4], "type": link[5]}
    return None


def _find_node(nodes: list, node_id) -> dict | None:
    for n in nodes:
        if str(n.get("id")) == str(node_id):
            return n
    return None


def _input_slot_index(node: dict, input_name: str, typ: str) -> int:
    """Index of *input_name* in node.inputs, promoting a widget to a link slot if
    the name isn't already a connectable input (so scalars/prompts can be exposed)."""
    inputs = node.setdefault("inputs", [])
    for i, s in enumerate(inputs):
        if s.get("name") == input_name:
            return i
    # Promote: add a widget-backed input slot so an external link can feed it.
    inputs.append({"name": input_name, "type": typ or "*",
                   "widget": {"name": input_name}, "link": None})
    return len(inputs) - 1


def _slot_type(node: dict, kind: str, key, fallback: str) -> str:
    """Resolve a slot's type from the inner node, falling back to *fallback*/'*'."""
    seq = node.get(kind) or []
    if kind == "inputs":
        for s in seq:
            if s.get("name") == key:
                return s.get("type") or fallback or "*"
    else:  # outputs, key is an index
        if isinstance(key, int) and 0 <= key < len(seq):
            return seq[key].get("type") or fallback or "*"
    return fallback or "*"


def build_subgraph_definition(stage: dict) -> dict:
    """Build one ``definitions.subgraphs`` entry from a graph-format *stage*.

    ``stage`` keys:
      - ``graph``   : graph-format workflow (``nodes`` + ``links``).
      - ``name``    : subgraph display name.
      - ``inputs``  : ``[{node_id, input, type?}]`` — inner inputs to expose, in
                      the order that must match the hook's input slots.
      - ``outputs`` : ``[{node_id, output_slot, type?}]`` — inner outputs to
                      expose, matching the hook's output slots.

    Returns the definition dict (with a fresh ``id`` uuid). The exposed
    ``inputs``/``outputs`` order defines the instance's slot order.
    """
    graph = copy.deepcopy(stage.get("graph") or {})
    nodes = graph.get("nodes") or []
    links = [lk for lk in (_as_object_link(l) for l in (graph.get("links") or [])) if lk]

    # Highest existing ids so boundary links / state counters don't collide.
    max_link = max([int(l["id"]) for l in links if isinstance(l.get("id"), int)] or [0])
    max_node = max([int(n["id"]) for n in nodes
                    if isinstance(n.get("id"), int) and n["id"] >= 0] or [0])

    exposed_inputs: list[dict] = []
    exposed_outputs: list[dict] = []

    # ── expose inputs: boundary link -10.i → inner (node, slot) ──
    for i, spec in enumerate(stage.get("inputs") or []):
        node = _find_node(nodes, spec.get("node_id"))
        if node is None:
            continue
        name = str(spec.get("input") or f"in{i}")
        typ = spec.get("type") or _slot_type(node, "inputs", name, "*")
        slot = _input_slot_index(node, name, typ)
        max_link += 1
        lid = max_link
        # Drop any pre-existing link into that inner slot; the boundary feeds it now.
        links = [l for l in links
                 if not (str(l["target_id"]) == str(node["id"]) and l["target_slot"] == slot)]
        node["inputs"][slot]["link"] = lid
        links.append({"id": lid, "origin_id": _INPUT_NODE_ID, "origin_slot": i,
                      "target_id": node["id"], "target_slot": slot, "type": typ})
        exposed_inputs.append({
            "id": _uuid(), "name": name, "type": typ, "linkIds": [lid],
            "localized_name": name, "label": name, "pos": [-260, 40 + i * 20],
        })

    # ── expose outputs: boundary link inner (node, slot) → -20.j ──
    for j, spec in enumerate(stage.get("outputs") or []):
        node = _find_node(nodes, spec.get("node_id"))
        if node is None:
            continue
        oslot = int(spec.get("output_slot", 0) or 0)
        outs = node.setdefault("outputs", [])
        while len(outs) <= oslot:
            outs.append({"name": f"OUT{len(outs)}", "type": "*", "links": None})
        typ = spec.get("type") or (outs[oslot].get("type") or "*")
        max_link += 1
        lid = max_link
        outs[oslot]["links"] = (outs[oslot].get("links") or []) + [lid]
        links.append({"id": lid, "origin_id": node["id"], "origin_slot": oslot,
                      "target_id": _OUTPUT_NODE_ID, "target_slot": j, "type": typ})
        nm = outs[oslot].get("name") or f"OUT{j}"
        exposed_outputs.append({
            "id": _uuid(), "name": nm, "type": typ, "linkIds": [lid],
            "localized_name": nm, "label": nm, "pos": [1200, 40 + j * 20],
        })

    # ── computed outputs: inject an AgentYPython node running the agent's runtime
    #    snippet, and expose its result as a subgraph output. This bakes a value the
    #    agent computed outside the graph (e.g. a video's length) into the native
    #    workflow, so re-running reproduces it without the agent. ──
    for spec in stage.get("computed_outputs") or []:
        code = str(spec.get("code") or "")
        max_node += 1
        py_id = max_node
        py_inputs: list[dict] = []
        for k, ref in enumerate(spec.get("inputs") or []):
            prod = _find_node(nodes, ref.get("node_id"))
            if prod is None:
                continue
            oslot = int(ref.get("output_slot", 0) or 0)
            outs = prod.setdefault("outputs", [])
            while len(outs) <= oslot:
                outs.append({"name": f"OUT{len(outs)}", "type": "*", "links": None})
            t = outs[oslot].get("type") or "*"
            max_link += 1
            lid = max_link
            outs[oslot]["links"] = (outs[oslot].get("links") or []) + [lid]
            links.append({"id": lid, "origin_id": prod["id"], "origin_slot": oslot,
                          "target_id": py_id, "target_slot": k, "type": t})
            py_inputs.append({"name": f"in{k}", "type": t, "link": lid})
        nodes.append({
            "id": py_id, "type": "AgentYPython",
            "pos": [1000, 300 + len(exposed_outputs) * 70], "size": [320, 180],
            "flags": {}, "order": 0, "mode": 0,
            "inputs": py_inputs,
            "outputs": [{"name": "out0", "type": "*", "links": None}],
            "properties": {"Node name for S&R": "AgentYPython"},
            "widgets_values": [code],
        })
        j = len(exposed_outputs)
        typ = spec.get("type") or "*"
        max_link += 1
        lid = max_link
        nodes[-1]["outputs"][0]["links"] = [lid]
        links.append({"id": lid, "origin_id": py_id, "origin_slot": 0,
                      "target_id": _OUTPUT_NODE_ID, "target_slot": j, "type": typ})
        nm = str(spec.get("name") or f"computed{j}")
        exposed_outputs.append({
            "id": _uuid(), "name": nm, "type": typ, "linkIds": [lid],
            "localized_name": nm, "label": nm, "pos": [1200, 40 + j * 20],
        })

    return {
        "id": _uuid(),
        "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": max_node,
                  "lastLinkId": max_link, "lastRerouteId": 0},
        "revision": 0,
        "config": {},
        "name": short_name(stage.get("name")),
        "inputNode": {"id": _INPUT_NODE_ID, "bounding": [-320, 0, 120, 60]},
        "outputNode": {"id": _OUTPUT_NODE_ID, "bounding": [1180, 0, 120, 60]},
        "inputs": exposed_inputs,
        "outputs": exposed_outputs,
        "widgets": [],
        "nodes": nodes,
        "groups": graph.get("groups") or [],
        "links": links,
        "extra": {},
    }


def _instance_node(inst_id: int, definition: dict, pos: list) -> dict:
    """A subgraph-instance node (``type`` = the definition's uuid) whose input/
    output slots mirror the definition's exposed I/O (names/types must match)."""
    ins = [{"name": s["name"], "type": s["type"], "link": None,
            "localized_name": s.get("localized_name", s["name"]),
            "label": s.get("label", s["name"])}
           for s in definition.get("inputs", [])]
    outs = [{"name": s["name"], "type": s["type"], "links": [],
             "localized_name": s.get("localized_name", s["name"]),
             "label": s.get("label", s["name"])}
            for s in definition.get("outputs", [])]
    return {
        "id": inst_id, "type": definition["id"], "pos": pos, "size": [260, 140],
        "flags": {}, "order": 0, "mode": 0, "inputs": ins, "outputs": outs,
        "properties": {}, "title": definition.get("name"),
    }


def build_baked_workflow(stages: list, links: list | None = None) -> dict:
    """Assemble a full graph-format canvas from ordered *stages*.

    Each stage is passed to :func:`build_subgraph_definition`; one instance node is
    created per stage and laid out left→right. ``links`` is an optional explicit
    wiring list ``[{from_stage, from_output, to_stage, to_input}]`` (stage = index
    into *stages*). When omitted, stages are chained linearly out0→in0.

    Returns a v0.4 graph-format workflow with ``definitions.subgraphs`` and the
    instance nodes, ready for ``loadGraphData`` / ``/agent/load_workflow``.
    """
    definitions: list[dict] = []
    instances: list[dict] = []
    for idx, stage in enumerate(stages or []):
        d = build_subgraph_definition(stage)
        definitions.append(d)
        instances.append(_instance_node(idx + 1, d, pos=[80 + idx * 380, 120]))

    # Wiring between instances.
    wiring = list(links or [])
    if not wiring and len(instances) > 1:
        # Default: linear chain, each stage's output 0 → next stage's input 0.
        for k in range(len(instances) - 1):
            wiring.append({"from_stage": k, "from_output": 0, "to_stage": k + 1, "to_input": 0})

    top_links: list = []
    lid = 0
    for w in wiring:
        try:
            a = instances[int(w["from_stage"])]
            b = instances[int(w["to_stage"])]
            osl = int(w.get("from_output", 0))
            isl = int(w.get("to_input", 0))
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if osl >= len(a["outputs"]) or isl >= len(b["inputs"]):
            continue
        lid += 1
        typ = a["outputs"][osl].get("type", "*")
        top_links.append([lid, a["id"], osl, b["id"], isl, typ])
        a["outputs"][osl]["links"] = (a["outputs"][osl].get("links") or []) + [lid]
        b["inputs"][isl]["link"] = lid

    return {
        "last_node_id": max([n["id"] for n in instances], default=0),
        "last_link_id": lid,
        "nodes": instances,
        "links": top_links,
        "groups": [],
        "config": {},
        "definitions": {"subgraphs": definitions},
        "extra": {},
        "version": 0.4,
    }
