"""Read the hook graph before it runs, and say what cannot work.

Every failure this file checks for was, at some point, discovered the expensive
way: after the prompts had been written, the references generated and the API
billed. They share a shape — the graph and the directives contradict each other,
and nothing looks at the two together until something downstream throws.

So this looks, before the turn starts, at what is actually knowable without
running anything: the captured graph after the hooks are spliced out, the hooks'
own declared target types, ComfyUI's node schemas, and the models' hard limits.
Three examples, all from one afternoon:

* a Kling node whose ``reference_images`` had nothing feeding it, because the
  hook that fed it was spliced out and no anchor could replace it;
* a hook feeding one slot of a batch node while its directive spoke of "all of
  the reference images";
* a directive addressing ``anchor_1`` when nothing was wired to that slot.

Findings come in two strengths. A **blocker** is something that will fail or
produce nothing — it is worth stopping for. A **note** is a mismatch between what
the graph can do and what the directive asks, which is usually a mistake and
occasionally deliberate; the agent is told, and decides.

Deliberately conservative: a check that fires on a working graph teaches people
to skip the block, which costs more than the check ever saved. Anything that
needs a running ComfyUI is best-effort and simply absent when it can't be asked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.canvas_hooks import (_COLLECTOR_TYPES, _hook_ids, _output_targets,
                                    _all_anchor_inputs, _slot_label,
                                    is_connection_type, missing_collector_files,
                                    unresolved_anchor_refs)

from src.utils.canvas_hooks import is_terminal


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the graph, as it stands."""

    level: str      # "blocker" | "note"
    hook: str       # the hook it belongs to, or "" for the graph as a whole
    text: str

    def line(self) -> str:
        who = f"hook {self.hook}: " if self.hook else ""
        return f"  {'BLOCKER' if self.level == 'blocker' else 'note'} — {who}{self.text}"


_schema_cache: dict = {}


def _schema(cls: str) -> dict:
    """One node's schema from ComfyUI, or {} when it can't be asked.

    Per class, not the whole database: ``/object_info`` is 1,700 nodes and nine
    seconds on this install, which is a lot to spend before a turn starts, while
    a single class is two hundred milliseconds and the graph only ever needs the
    handful of nodes the hooks touch. Failures are cached too — a ComfyUI that is
    down should cost one refused connection, not one per node.
    """
    cls = str(cls or "")
    if not cls:
        return {}
    if cls in _schema_cache:
        return _schema_cache[cls]
    out: dict = {}
    try:
        from agenty_core.utils.comfyui_client import get_client
        data = get_client().get(f"/object_info/{cls}")
        if isinstance(data, dict):
            out = data.get(cls) or (next(iter(data.values()), {}) if len(data) == 1 else {})
    except Exception:  # noqa: BLE001
        out = {}
    _schema_cache[cls] = out if isinstance(out, dict) else {}
    return _schema_cache[cls]


def _required_inputs(schema: dict) -> dict:
    """``{name: declared_type}`` for a node's required inputs, connections only.

    Widgets are skipped: a required INT with a default is filled by ComfyUI, and
    reporting it as missing would fire on every healthy graph.
    """
    out: dict = {}
    for name, spec in ((schema.get("input") or {}).get("required") or {}).items():
        t = spec[0] if isinstance(spec, list) and spec else spec
        if isinstance(t, str) and is_connection_type(t):
            out[name] = t
    return out


def _reaches_output(prompt: dict) -> bool:
    """Whether anything in the graph saves, previews or displays a result.

    One definition of "output node", shared with the pruning in
    :mod:`canvas_hooks` — the name is tried first and answers for almost every
    graph without asking ComfyUI anything, and a class it cannot ask about counts
    as an output, so a ComfyUI that is down never produces "nothing renders".
    """
    return any(is_terminal(n.get("class_type")) for n in (prompt or {}).values()
               if isinstance(n, dict))


def _anchor_types(hook: dict, base_prompt: dict | None) -> set:
    """The output types wired into a hook, as far as they were reported."""
    out: set = set()
    for a in (hook.get("anchors") or []):
        if isinstance(a, dict):
            t = str(a.get("from_output_type") or "").strip()
            if t:
                out.add(t)
    return out


def _wired_slots(hook: dict) -> list:
    """The anchor slot NAMES this hook actually has something on, in slot order.

    Names, not numbers: the slots are ``anchor``, ``anchor0``, ``anchor1``, … so
    the first has no number at all, and folding that to ``0`` made it collide with
    the real ``anchor0``. A hook with two inputs then reported ONE wired slot, and
    a directive mentioning its second input was flagged as pointing at nothing.
    """
    names: list = []
    for src in ("anchors", "prev_links"):
        for item in (hook.get(src) or []):
            if isinstance(item, dict):
                label = _slot_label(str(item.get("to_input") or ""))
                if label and label not in names:
                    names.append(label)
    return names


def check(hooks: list | None, base_prompt: dict | None) -> list:
    """Everything wrong with this hook graph that can be known before it runs."""
    hooks = [h for h in (hooks or []) if isinstance(h, dict)]
    if not hooks:
        return []
    found: list = []
    ids = _hook_ids(hooks)
    prompt = base_prompt if isinstance(base_prompt, dict) else {}

    # ── the graph as a whole ────────────────────────────────────────────────
    if prompt and not _reaches_output(prompt):
        found.append(Finding("blocker", "", (
            "nothing in this graph saves, previews or displays anything, so a run "
            "produces no files. Ask the user to wire an output node (SaveImage, a "
            "viewer) to the node that generates.")))

    # Only the nodes the hooks touch, and what those feed. Splicing is what leaves
    # an input unfilled, so that is where the check belongs — and it keeps this to
    # a handful of schema lookups rather than one per node in the graph.
    touched: set = {str(t[0]) for h in hooks for t in _output_targets(h)}
    for nid in sorted(touched):
        node = prompt.get(nid)
        if not isinstance(node, dict) or nid in ids:
            continue
        cls = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        pending = {t[2] for h in hooks for t in _output_targets(h) if str(t[0]) == nid}
        for name, declared in _required_inputs(_schema(cls)).items():
            # A hook about to fill it is not missing, it is pending.
            if name in inputs or name in pending:
                continue
            found.append(Finding("blocker", "", (
                f"node {nid} ({cls}) needs `{name}` ({declared}) and nothing feeds "
                f"it — the run fails validation before it starts. Usually the hook "
                f"that fed it was spliced out and no anchor could replace it.")))

    for graph_check in (missing_collector_files(prompt) or []):
        found.append(Finding("note", "", (
            f"node {graph_check['node_id']} ({graph_check['class_type']}) lists "
            f"{len(graph_check['missing'])} of {graph_check['lines']} file(s) that do "
            f"not exist: {'; '.join(graph_check['missing'][:4])}. The collector skips "
            f"what it cannot find, so the run uses fewer images than the list says.")))

    # ── per hook ────────────────────────────────────────────────────────────
    for h in hooks:
        hid = str(h.get("hook_node_id"))
        directive = str(h.get("directive") or "")
        targets = _output_targets(h)
        anchors = _all_anchor_inputs(h, prompt)
        have = _anchor_types(h, prompt)

        # A directive naming a slot no reading can reach. "anchor_1" means either
        # the slot called anchor1 or the first input wired — the node's own naming
        # (anchor, anchor0, anchor1) makes both reasonable, so a check that picks
        # one warns about healthy graphs. Only a number beyond every reading is a
        # finding; see canvas_hooks.anchor_slot_matches.
        wired = _wired_slots(h)
        named = {int(m.group(1)) for m in re.finditer(r"anchor[_ \-]?(\d+)", directive)}
        for slot in unresolved_anchor_refs(named, wired):
            found.append(Finding("note", hid, (
                f"the directive refers to anchor_{slot}, but this hook has "
                f"{len(wired)} input(s) wired ({', '.join(wired) or 'none'}) and no "
                f"reading of that name reaches one. Whatever it expects there, it "
                f"will not find.")))

        # Two targets that want different things. This is a NOTE, not a blocker:
        # it is only unsatisfiable on the place_canvas_text path, which sends one
        # string to every target. apply_canvas_hooks takes a resolution PER target,
        # so the same hook can legitimately fill a collector's `files` with paths
        # and a prompt box with prose. Calling it a blocker stopped a working run
        # and made the agent report a defect that was not there — a check that
        # halts a good graph costs more than the check ever saved.
        wants_paths, wants_prose = [], []
        for tid, _ttype, tin, tintype, ttitle in targets:
            if is_connection_type(tintype) or not tin:
                continue
            cls = str((prompt.get(str(tid)) or {}).get("class_type") or _ttype or "")
            if cls in _COLLECTOR_TYPES or tin in ("files", "paths", "file_list"):
                wants_paths.append(f"node {tid}'s `{tin}`")
            elif re.search(r"prompt|text|caption|description", tin, re.I):
                wants_prose.append(f"node {tid}'s `{tin}`")
        if wants_paths and wants_prose:
            found.append(Finding("note", hid, (
                f"its output feeds {', '.join(wants_paths)} — which needs absolute file "
                f"paths, one per line — AND {', '.join(wants_prose)}, which needs a "
                f"written prompt. These want different content, so do NOT deliver this "
                f"hook with place_canvas_text: that sends the same string to every "
                f"target and one of them would get the other's value. Use "
                f"apply_canvas_hooks with one resolution per target instead.")))

        for tid, _ttype, tin, tintype, _ttitle in targets:
            if not tin or not is_connection_type(tintype) or str(tid) in ids:
                continue
            # A connection target this hook has nothing to feed with.
            if have and tintype not in have and "*" not in have and not (
                    have & {"COMFY_MATCHTYPE_V3", "COMFY_MULTITYPE_V3"}):
                found.append(Finding("note", hid, (
                    f"its output feeds node {tid}'s `{tin}` ({tintype}), but nothing "
                    f"wired into this hook produces a {tintype} "
                    f"(it has: {', '.join(sorted(have))}). You can still name a node id "
                    f"or a file for it, but there is no anchor to choose from.")))
            # One slot of an autogrow input, several images on the hook.
            if re.search(r"\d$", tin) and len(anchors) > 1 and tintype == "IMAGE":
                sibling = re.sub(r"\d+$", "", tin)
                others = [k for k in ((prompt.get(str(tid)) or {}).get("inputs") or {})
                          if k.startswith(sibling) and k != tin]
                if not others:
                    found.append(Finding("note", hid, (
                        f"its output feeds ONE image slot (node {tid}'s `{tin}`) while "
                        f"{len(anchors)} inputs are wired into the hook, and no sibling "
                        f"slot (`{sibling}2`, …) is wired on that node. A sweep fills one "
                        f"slot once per RUN, so as it stands each run sees a single "
                        f"image. `{tin}` is a numbered slot, so the node most likely "
                        f"grows more as they are wired — if these references belong in "
                        f"ONE generation, wire the extra slots (or an agentY image "
                        f"collector) rather than sweeping them.")))

    return found


def lines(hooks: list | None, base_prompt: dict | None) -> list:
    """The PRE-FLIGHT block, or [] when the graph has nothing wrong with it."""
    found = check(hooks, base_prompt)
    if not found:
        return []
    blockers = [f for f in found if f.level == "blocker"]
    out = ["\nPRE-FLIGHT (checked against the graph and the node schemas, before "
           "anything ran):"]
    out += [f.line() for f in blockers] + [f.line() for f in found if f.level != "blocker"]
    if blockers:
        out.append("  A BLOCKER will fail or produce nothing. Do not start the run: say "
                   "which one and what to wire, and let the user fix it — unless they "
                   "have already told you to proceed anyway.")
    return out
