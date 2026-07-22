"""Helpers for agentY canvas-hook nodes.

A canvas ``AgentYHook`` node lets the user annotate any node's output with a
natural-language directive ("sweep the seed", "iterate this folder"). On a normal
ComfyUI run the hook is inert. When the agent runs the on-canvas graph, the
frontend ships the API-format prompt plus the hook directives; these helpers:

* ``splice_hook_nodes`` — remove the hook nodes from the API prompt (rewiring any
  inline hook so the graph stays connected), yielding the clean base prompt.
* ``build_batch`` — expand the base prompt into a mutated batch (seed sweep,
  value list, folder iterate), as the capped Cartesian product of the per-node
  resolutions the orchestrator supplies.
* ``describe_hooks`` — render the ``[CANVAS HOOKS]`` block injected into the
  orchestrator's input so it knows what to expand.
"""
from __future__ import annotations

import copy
import random
import re
from pathlib import Path

_HOOK_CLASS = "AgentYHook"

IMG_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"}
VID_EXTS = {"mp4", "mov", "webm", "mkv", "avi"}


def _anchor_links(inputs: dict) -> list:
    """Return each wired ``anchor`` link ``[node_id, slot]`` on a hook, in slot
    order. The anchor input auto-grows, so its names are ``anchor``/``anchor0``/… .
    The V3 Autogrow schema prefixes them with the group name (``anchors.anchor0``),
    so match the trailing ``anchorN`` after an optional ``group.`` prefix — else the
    rewire in ``splice_hook_nodes`` disconnects inline hooks instead of wiring them."""
    def _tail(name: str) -> str:
        return name.rsplit(".", 1)[-1]  # drop any "anchors." group prefix

    def _idx(name: str) -> int:
        suf = _tail(name)[len("anchor"):]
        return int(suf) if suf.isdigit() else -1  # bare "anchor" sorts first

    def _is_anchor(k: str) -> bool:
        t = _tail(k)
        return t == "anchor" or (t.startswith("anchor") and t[len("anchor"):].isdigit())

    keys = [k for k in (inputs or {}) if _is_anchor(k)]
    out = []
    for k in sorted(keys, key=_idx):
        v = inputs.get(k)
        if isinstance(v, list) and len(v) == 2:
            out.append(v)
    return out


def splice_hook_nodes(prompt: dict) -> tuple[dict, list[str]]:
    """Return ``(clean_prompt, removed_ids)``.

    Removes every ``AgentYHook`` node from an API-format *prompt*. For an inline
    hook (its output feeds a downstream node) each downstream input is rewired to
    the hook's own ``anchor`` source so the graph stays connected. The anchor
    input auto-grows (``anchor``, ``anchor0``, ``anchor1``, …), so a hook may have
    several wired inputs; the first (lowest-slot) is used for the rewire, matching
    the node's passthrough. A dangling hook (nothing consumes it) is simply
    dropped. Works whether or not ComfyUI's ``graphToPrompt`` already pruned the
    hook.
    """
    if not isinstance(prompt, dict):
        return prompt, []
    clean = copy.deepcopy(prompt)
    hook_ids = [nid for nid, node in clean.items()
                if isinstance(node, dict) and node.get("class_type") == _HOOK_CLASS]
    if not hook_ids:
        return clean, []
    for hid in hook_ids:
        node = clean.get(hid, {}) or {}
        src = next(iter(_anchor_links(node.get("inputs") or {})), None)
        # Rewire any consumer of this hook's output back to the hook's source.
        for other in clean.values():
            if not isinstance(other, dict):
                continue
            for k, v in list((other.get("inputs") or {}).items()):
                if isinstance(v, list) and len(v) == 2 and str(v[0]) == str(hid):
                    if src is not None:
                        other["inputs"][k] = list(src)
                    else:
                        other["inputs"].pop(k, None)
        clean.pop(hid, None)
    return clean, hook_ids


def _rand_seed() -> int:
    return random.randint(0, 2**31 - 1)


def enumerate_folder(folder: str, extensions=None, use_full_path: bool = False) -> list[str]:
    """List files in *folder* (sorted), optionally filtered by extension."""
    p = Path(folder)
    if not p.is_dir():
        return []
    exts = {e.lower().lstrip(".") for e in (extensions or [])}
    out: list[str] = []
    for f in sorted(p.iterdir()):
        if not f.is_file():
            continue
        if exts and f.suffix.lower().lstrip(".") not in exts:
            continue
        out.append(str(f) if use_full_path else f.name)
    return out


def _resolve_values(res: dict) -> list:
    """Turn one resolution spec into the concrete list of values to sweep."""
    mode = str(res.get("mode", "value_list") or "value_list")
    if mode in ("sweep_seed", "seed", "seed-sweep"):
        count = int(res.get("count", 4) or 4)
        start = res.get("start")
        if start is not None:
            try:
                start = int(start)
                return [start + i for i in range(count)]
            except (TypeError, ValueError):
                pass
        return [_rand_seed() for _ in range(count)]
    if mode in ("folder", "folder_iterate", "file-iterate"):
        return enumerate_folder(
            str(res.get("folder", "") or ""),
            res.get("extensions"),
            bool(res.get("use_full_path", False)),
        )
    return list(res.get("values") or [])


def _extract_key(value, pattern: str) -> str | None:
    """Derive a join key from one file *value* for name-based zipping. The key is the
    basename stem (extension dropped); if *pattern* is given, it's a regex searched in
    the stem — the first capture group if any, else the whole match. Returns None when
    nothing matches (that value is then unmatched and dropped from the join)."""
    stem = Path(str(value)).stem
    if not pattern:
        return stem or None
    try:
        m = re.search(pattern, stem)
    except re.error:
        return None
    if not m:
        return None
    return m.group(1) if m.groups() else m.group(0)


def _is_join_key_member(res: dict) -> bool:
    """A ``mode: "join_key"`` member carries no values of its own — in a name-joined
    group it receives the shared shot key as its value (e.g. a save node's
    ``filename_prefix``), so each output is named by the key it was paired on."""
    return str(res.get("mode", "") or "").strip().lower() in ("join_key", "key")


def _resolve_group(members: list) -> tuple[list, list[str]]:
    """Resolve one zip group into a list of *rows*, each a list of
    ``(node_id, param, value)`` assignments applied together. A single non-join member
    is an ordinary product axis (one row per value). Two+ members advance TOGETHER —
    positionally by index (default), or joined on a filename shot-key when any member
    sets ``match_by: "name"`` (optionally with a ``key_pattern`` regex)."""
    notes: list[str] = []

    # A lone ordinary axis — the Cartesian-product path, unchanged from before.
    value_members = [(n, p, r) for (n, p, r) in members if not _is_join_key_member(r)]
    join_key_members = [(n, p, r) for (n, p, r) in members if _is_join_key_member(r)]
    if len(members) == 1 and not join_key_members:
        nid, param, res = members[0]
        values = _resolve_values(res)
        if not values:
            return [], [f"node {nid}.{param}: no values resolved — skipped"]
        return [[(nid, param, v)] for v in values], notes
    if not value_members:
        return [], ["zip group has no value members — skipped"]

    resolved = [(nid, param, res, _resolve_values(res)) for (nid, param, res) in value_members]
    for nid, param, _res, vals in resolved:
        if not vals:
            notes.append(f"node {nid}.{param}: no values resolved — zip group skipped")
            return [], notes

    name_join = any(
        str(res.get("match_by", "") or "").strip().lower() in ("name", "key", "filename")
        or str(res.get("key_pattern", "") or "").strip()
        for (_n, _p, res, _v) in resolved
    )

    if not name_join:
        # Positional zip — advance value lists in lockstep by index.
        lengths = [len(v) for (_n, _p, _r, v) in resolved]
        n = min(lengths)
        if max(lengths) != n:
            notes.append(f"positional zip: lists differ in length {lengths}; using the first {n}")
        if join_key_members:
            notes.append("join_key member needs match_by=name (no key in a positional zip) — skipped")
        rows = [[(nid, param, vals[i]) for (nid, param, _r, vals) in resolved] for i in range(n)]
        return rows, notes

    # Name-key join — pair values across members by a shot key from their filenames.
    keyed: list = []
    for nid, param, res, vals in resolved:
        pat = str(res.get("key_pattern", "") or "").strip()
        km: dict = {}
        for v in vals:
            k = _extract_key(v, pat)
            if k is None:
                notes.append(f"node {nid}.{param}: no key from {v!r} (pattern {pat!r}) — skipped")
                continue
            if k not in km:
                km[k] = v
            else:
                notes.append(f"node {nid}.{param}: duplicate key {k!r} — keeping the first")
        keyed.append((nid, param, km))

    common: set | None = None
    all_keys: set = set()
    for _n, _p, km in keyed:
        all_keys |= set(km)
        common = set(km) if common is None else (common & set(km))
    common_sorted = sorted(common or [])
    dropped = sorted(all_keys - set(common_sorted))
    if dropped:
        shown = ", ".join(dropped[:10]) + (" …" if len(dropped) > 10 else "")
        notes.append(f"name-join: {len(dropped)} unmatched key(s) skipped: {shown}")
    if not common_sorted:
        notes.append("name-join matched 0 keys — check key_pattern / filenames")
        return [], notes

    rows = []
    for k in common_sorted:
        row = [(nid, param, km[k]) for (nid, param, km) in keyed]
        row += [(nid, param, k) for (nid, param, _res) in join_key_members]
        rows.append(row)
    return rows, notes


def build_batch(base_prompt: dict, resolutions: list, cap: int = 25) -> tuple[list[dict], list[str]]:
    """Expand *base_prompt* into a mutated batch from *resolutions*.

    Each resolution mutates one node's input across a list of values. By default the
    batch is the **Cartesian product** across resolutions. Resolutions sharing a
    non-empty ``zip_group`` advance **together** instead of crossing — a *zip*: by
    list index (default), or joined on a filename shot-key when a member sets
    ``match_by: "name"`` (optionally with a ``key_pattern`` regex). A ``mode:
    "join_key"`` member in a name-joined group receives that shared key as its value
    (e.g. to name each output). Groups (and ungrouped axes) then cross-product with
    each other. Capped at *cap*. Returns ``(prompts, notes)`` where *notes* explains
    any skips/truncation.
    """
    notes: list[str] = []
    # Bucket resolutions into groups, preserving encounter order. Each ungrouped
    # resolution is its own singleton group (a plain product axis, as before).
    groups: dict[str, list] = {}
    order: list[str] = []
    solo = 0
    for res in (resolutions or []):
        if not isinstance(res, dict):
            continue
        nid = str(res.get("target_node_id", res.get("node_id", "")) or "")
        param = str(res.get("param", "") or "")
        if not nid or not param:
            notes.append(f"skipped a resolution missing target_node_id/param: {res!r}")
            continue
        if nid not in base_prompt:
            notes.append(f"node {nid} is not in the canvas graph — skipped")
            continue
        gid = str(res.get("zip_group", "") or "").strip()
        if not gid:
            gid = f"\x00solo{solo}"
            solo += 1
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append((nid, param, res))

    group_rows: list[list] = []
    for gid in order:
        rows, gnotes = _resolve_group(groups[gid])
        notes.extend(gnotes)
        if rows:
            group_rows.append(rows)

    if not group_rows:
        return [], (notes or ["no valid resolutions were supplied"])

    combos: list[list] = [[]]
    for rows in group_rows:
        combos = [c + [r] for c in combos for r in rows]
    total = len(combos)
    if total > cap:
        notes.append(f"batch of {total} exceeded the cap of {cap}; truncated to {cap}")
        combos = combos[:cap]

    prompts: list[dict] = []
    for combo in combos:
        p = copy.deepcopy(base_prompt)
        for row in combo:                       # each row is one group's aligned assignments
            for (nid, param, val) in row:
                node = p.get(nid)
                if isinstance(node, dict):
                    node.setdefault("inputs", {})[param] = val
        prompts.append(p)
    return prompts, notes


# ``make_workflow`` is the current purpose name; the older ``workflow-standin``
# spellings are kept so canvases saved before the rename still resolve correctly.
_STANDIN_PURPOSES = {"make_workflow", "make-workflow", "workflow-standin",
                     "workflow_standin", "standin", "workflow"}
_TEXT_PURPOSES = {"text", "text-output", "text_output", "answer"}


def _is_standin(hook: dict) -> bool:
    """True if *hook* is a make_workflow hook (vs. an inline_parameter annotation)."""
    return str(hook.get("purpose", "inline_parameter") or "inline_parameter").strip().lower() in _STANDIN_PURPOSES


def _is_text(hook: dict) -> bool:
    """True if *hook* asks for a written text answer (no media, no workflow)."""
    return str(hook.get("purpose", "inline_parameter") or "inline_parameter").strip().lower() in _TEXT_PURPOSES


_ITERATE_PURPOSES = {"iterate", "iterative", "refine", "loop",
                     "iterative_refine", "iterate_loop", "refine_loop"}


def _is_iterate(hook: dict) -> bool:
    """True if *hook* declares an interactive iterative-refinement loop — one
    generation per turn, feeding each result back into the wired LoadImage node.
    Driven by the ``iterate_step`` tool + the ``iterative-refine`` skill, not by
    the one-shot producer/standin paths."""
    return str(hook.get("purpose", "") or "").strip().lower() in _ITERATE_PURPOSES


def _wants_bake(hook: dict) -> bool:
    """True if *hook* has the ``bake_to_canvas`` switch on (bake to a subgraph)."""
    v = hook.get("bake")
    return v is True or str(v).strip().lower() in ("true", "1", "yes", "on")


def _export_count(hook: dict) -> int:
    """How many outputs this standin should export (wired output slots, ≥1)."""
    n = int(hook.get("outputs_wired") or 0) or int(hook.get("output_count") or 0)
    return max(n, 1)


def _order_standin_chains(standin_hooks: list) -> list:
    """Group standin hooks into ordered chains via their ``prev_hook_id`` links.

    A hook wired FROM another standin hook is a downstream stage; ``prev_hook_id``
    names its predecessor. Returns a list of chains, each an ordered list of hooks
    (a standalone standin is a chain of length 1). Heads are hooks whose
    predecessor is absent or not itself a standin; successors are followed one at
    a time (a fork just starts a new chain), and cycles are broken defensively.
    """
    by_id = {str(h.get("hook_node_id")): h for h in standin_hooks if h.get("hook_node_id") is not None}
    # next_of[pred_id] = successor hook (first one wins if a stage forks).
    next_of: dict = {}
    for h in standin_hooks:
        prev = h.get("prev_hook_id")
        if prev is not None and str(prev) in by_id and str(prev) not in next_of:
            next_of[str(prev)] = h

    def _is_head(h: dict) -> bool:
        prev = h.get("prev_hook_id")
        return prev is None or str(prev) not in by_id

    chains: list = []
    seen: set = set()
    for h in standin_hooks:
        if not _is_head(h) or str(h.get("hook_node_id")) in seen:
            continue
        chain, cur = [], h
        while cur is not None and str(cur.get("hook_node_id")) not in seen:
            seen.add(str(cur.get("hook_node_id")))
            chain.append(cur)
            cur = next_of.get(str(cur.get("hook_node_id")))
        chains.append(chain)
    # Any hooks left unvisited (e.g. caught in a cycle) become their own chains.
    for h in standin_hooks:
        if str(h.get("hook_node_id")) not in seen:
            seen.add(str(h.get("hook_node_id")))
            chains.append([h])
    return chains


def _all_anchor_inputs(hook: dict, base_prompt: dict | None) -> list:
    """Return ``[(anchor_id, anchor_type, scalar_inputs_dict), …]`` for every
    real-node input wired to a hook.

    The anchor input auto-grows, so a hook may gather several inputs (carried in
    the ``anchors`` list). Falls back to the singular ``anchor_node_id`` field for
    older frontends that only send one.
    """
    entries: list = []
    plural = hook.get("anchors")
    if isinstance(plural, list) and plural:
        for a in plural:
            if isinstance(a, dict) and a.get("node_id") is not None:
                entries.append((str(a["node_id"]), a.get("type"), a.get("widgets")))
    elif hook.get("anchor_node_id") is not None:
        entries.append((str(hook["anchor_node_id"]), hook.get("anchor_type"),
                        hook.get("anchor_widgets")))

    out: list = []
    seen: set = set()
    for aid, atype, widgets in entries:
        if aid in seen:
            continue
        seen.add(aid)
        inputs: dict = {}
        if base_prompt and aid in base_prompt:
            raw = (base_prompt[aid].get("inputs") or {})
            inputs = {k: v for k, v in raw.items() if not isinstance(v, list)}
        elif isinstance(widgets, dict):
            inputs = widgets
        out.append((aid, atype or "?", inputs))
    return out


def _output_targets(hook: dict) -> list:
    """Return ``[(target_id, target_type, to_input, to_input_type, target_title), …]``
    for every node input this hook's output is wired into — the producer's
    DESTINATION(s). A hook is an upstream producer: it consumes its anchor inputs
    as context and produces value(s) for its ``out``, which the user wires into a
    real input. Knowing the exact target lets the agent fill/sweep the RIGHT input
    (derived from the wire) instead of guessing "the connected node" from prose.
    Empty when the hook's output is unwired.
    """
    out: list = []
    seen: set = set()
    for t in (hook.get("targets") or []):
        if not (isinstance(t, dict) and t.get("node_id") is not None):
            continue
        key = (str(t["node_id"]), str(t.get("to_input") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append((str(t["node_id"]), str(t.get("type") or "?"),
                    str(t.get("to_input") or ""), str(t.get("to_input_type") or ""),
                    str(t.get("title") or "")))
    return out


def inject_produced_value(base_prompt: dict, hook: dict, value) -> list[str]:
    """Write a producer hook's single produced *value* into each REAL node input
    its output feeds, returning the target ids actually written.

    This is the **keep-live** delivery path (the alternative to baking + rewiring an
    ``agentY text`` node into the target): the hook stays wired exactly as the user
    drew it, but the base graph the server queues carries *value* at the wired
    input — so a normal server-side run renders it without touching the canvas.
    Targets not present in *base_prompt* (e.g. another hook consumed only as
    context) are skipped, matching how ``build_batch`` guards its sweeps.
    """
    written: list[str] = []
    if not isinstance(base_prompt, dict):
        return written
    for tid, _ttype, tin, _tintype, _ttitle in _output_targets(hook):
        node = base_prompt.get(tid)
        if not isinstance(node, dict) or not tin:
            continue
        node.setdefault("inputs", {})[tin] = value
        written.append(tid)
    return written


def _hook_ids(hooks: list) -> set:
    """The set of node ids that are themselves hooks (for chain detection)."""
    return {str(h.get("hook_node_id")) for h in hooks if h.get("hook_node_id") is not None}


def _hook_predecessors(hook: dict, hook_ids: set) -> set:
    """Ids of hooks whose output feeds one of *hook*'s inputs (its producers).

    A hook depends on another when the latter is wired into one of its anchors
    (``anchors[].node_id`` is a hook id) or recorded in ``prev_hook_ids``. Those
    producers must run first so their value exists when this hook is processed.
    """
    ids: set = {str(p) for p in (hook.get("prev_hook_ids") or []) if str(p) in hook_ids}
    for a in (hook.get("anchors") or []):
        if isinstance(a, dict):
            aid = str(a.get("node_id"))
            if aid in hook_ids:
                ids.add(aid)
    if hook.get("prev_hook_id") is not None and str(hook["prev_hook_id"]) in hook_ids:
        ids.add(str(hook["prev_hook_id"]))
    ids.discard(str(hook.get("hook_node_id")))
    return ids


def _order_by_dependency(hooks: list) -> list:
    """Order *hooks* so every producer precedes the hook(s) consuming its output.

    A depth-first topological sort over the hook→hook wiring; input order is the
    tie-breaker and any cycle is broken defensively (a hook already on the stack
    is not revisited). This is what lets a producer bake its value before the
    consumer reads it — the sequencing comes from the graph, not a live re-snapshot.
    """
    ids = _hook_ids(hooks)
    by_id = {str(h.get("hook_node_id")): h for h in hooks if h.get("hook_node_id") is not None}
    ordered: list = []
    seen: set = set()

    def visit(h: dict, stack: set) -> None:
        hid = str(h.get("hook_node_id"))
        if hid in seen or hid in stack:
            return
        stack.add(hid)
        for pid in _hook_predecessors(h, ids):
            if pid in by_id:
                visit(by_id[pid], stack)
        stack.discard(hid)
        if hid not in seen:
            seen.add(hid)
            ordered.append(h)

    for h in hooks:
        visit(h, set())
    return ordered


# agentY collector nodes gather a set of on-disk files (image / video) into their
# ``files`` widget. Because that list is plain node data, the paths are known to
# the agent BEFORE any run — so an anchored collector is rendered as its explicit
# file list (not an opaque widget dump), telling the agent it can use every path
# directly without executing the graph first.
_COLLECTOR_TYPES = {"AgentYImageCollector", "AgentYVideoCollector"}


def _render_anchor(aid: str, atype: str, inputs: dict) -> str:
    """Human-readable description of one real-node anchor input. An agentY
    collector node is expanded to its listed on-disk file paths (available with no
    pre-run) so the agent can see/bind each file; other nodes list scalar params."""
    if atype in _COLLECTOR_TYPES:
        files = inputs.get("files") if isinstance(inputs, dict) else None
        paths = [ln.strip().strip('"') for ln in str(files or "").splitlines() if ln.strip()]
        kind = "image" if atype == "AgentYImageCollector" else "video"
        if not paths:
            return f"node {aid} (agentY {kind} collector) — EMPTY (no files added yet)"
        return (f"node {aid} (agentY {kind} collector) — {len(paths)} {kind} file(s) already "
                f"on disk (use these paths directly, no run needed): " + "; ".join(paths))
    params = ", ".join(f"{k}={v!r}" for k, v in (inputs or {}).items()) or "(no scalar inputs)"
    return f"node {aid} ({atype}) inputs[{params}]"


def _input_context(hook: dict, base_prompt: dict | None, hook_ids: set) -> str:
    """Describe what feeds *hook* (its context inputs).

    An input wired from another HOOK is rendered as "the value you produce for
    hook N" — never a dump of that hook's own widgets — so a chained producer
    reuses the value it just wrote instead of mistaking the upstream hook's
    directive for content. Real-node inputs list their scalar params (or, for an
    agentY collector, its explicit file list).
    """
    anchors = _all_anchor_inputs(hook, base_prompt)
    if not anchors:
        return "no input wired"
    parts: list = []
    for aid, atype, inputs in anchors:
        if aid in hook_ids:
            parts.append(f"the value you produce for hook {aid}")
        else:
            parts.append(_render_anchor(aid, atype, inputs))
    return "; ".join(parts)


def _target_context(hook: dict) -> str:
    """Describe where *hook*'s output goes — the producer's destination input(s)."""
    targets = _output_targets(hook)
    if not targets:
        return ""
    parts: list = []
    for tid, ttype, tin, tintype, _ttitle in targets:
        tt = f", {tintype}" if tintype else ""
        slot = f"`{tin}`" if tin else "an input"
        parts.append(f"node {tid} ({ttype})'s {slot} input{tt}")
    return "; ".join(parts)


def describe_hooks(hooks: list, base_prompt: dict | None = None) -> str:
    """Render the ``[CANVAS HOOKS]`` block injected into the orchestrator input.

    Hooks are **upstream producers**: each consumes its wired anchor inputs as
    context and produces value(s) for its ``out``, which the user wires into a real
    node input. Three purposes: an *inline_parameter* (producer) hook fills (or
    sweeps) the input its output is wired to; a *text* hook writes a single string
    the agent bakes there as an ``agentY text`` node; a *make_workflow* hook stands
    in for a workflow/script the agent generates. Hooks the user toggled to *ignore* are
    filtered out client-side, so every hook below is active. Hooks are described in
    dependency order so a producer is handled before the hook that consumes it.
    """
    hooks = [h for h in (hooks or []) if isinstance(h, dict)]
    if not hooks:
        return ""
    hooks = _order_by_dependency(hooks)
    hook_id_set = _hook_ids(hooks)
    text_hooks = [h for h in hooks if _is_text(h)]
    standin_hooks = [h for h in hooks if _is_standin(h)]
    iterate_hooks = [h for h in hooks if _is_iterate(h)]
    directive_hooks = [h for h in hooks
                       if not _is_standin(h) and not _is_text(h) and not _is_iterate(h)]

    lines = [
        "[CANVAS HOOKS — the user's ON-CANVAS graph carries hook annotations (below) "
        "and is already captured. Each hook is an UPSTREAM PRODUCER: it reads its "
        "wired anchor input(s) as context and produces value(s) for its output, which "
        "is wired into a real node input. Your job is to PRODUCE those values — fill "
        "or sweep the input each hook's output feeds — not to reach downstream and "
        "guess which node to edit. IF the user is asking you to run/execute the "
        "workflow, act on the hooks below. If the user's message is unrelated (a "
        "question or a different request), answer that and ignore these hooks.]"
    ]

    # When hooks feed each other, spell out the order so producers are done first.
    if any(_hook_predecessors(h, hook_id_set) for h in hooks):
        order = " → ".join(str(h.get("hook_node_id")) for h in hooks)
        lines.append(
            f"\nPROCESS ORDER (producers first): {order}. A hook whose input is 'the "
            "value you produce for hook N' consumes another hook's output — produce hook "
            "N FIRST and reuse exactly what you wrote as this hook's context; do NOT "
            "re-read it from the graph."
        )

    if directive_hooks:
        lines.append(
            "\nPRODUCER hooks — each produces value(s) for the node input its OUTPUT is "
            "wired to (shown as 'feeds …' below); that wired input is the target — do NOT "
            "guess a node from the prose, and do NOT assemble a template or call "
            "run_research. Two ways to produce, by how many values the directive asks for:\n"
            "  • ONE value (e.g. a single composed prompt) → write it and call "
            'place_canvas_text(hook_node_id="<hook id>", text="<value>") — it delivers the '
            "value to the target input (injected at run time if the hook is kept live, or "
            "baked in if frozen — the hook's own setting) and drops an 'agentY text' node.\n"
            "  • SEVERAL values (a sweep/variations/folder) → call "
            "apply_canvas_hooks(resolutions=[…]) ONCE with target_node_id + param taken "
            "straight from the 'feeds' target (node id and input name); each variant runs "
            "automatically — do NOT also signal_workflow_ready. Modes: value_list "
            "(you author `values`), sweep_seed (`count`, optional `start`), folder "
            "(`folder`, optional `extensions`)."
        )
        for h in directive_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            ctx = _input_context(h, base_prompt, hook_id_set)
            tgt = _target_context(h)
            if not tgt:
                lines.append(
                    f'- PRODUCER hook {hid} (context: {ctx}) — OUTPUT UNWIRED: no target '
                    f'input. Ask the user to wire this hook\'s output into the node input it '
                    f'should fill. Directive: "{directive}"'
                )
            else:
                lines.append(
                    f'- PRODUCER hook {hid} (context: {ctx}) feeds {tgt} — produce the '
                    f'value(s) for that input → "{directive}"'
                )

    if iterate_hooks:
        lines.append(
            "\nITERATIVE-REFINE hook(s) — the user wants an INTERACTIVE refinement LOOP on "
            "this on-canvas graph: ONE generation per turn, feeding each result back in as "
            "the next input. Activate the `iterative-refine` skill and follow it. Each turn, "
            "take the user's requested prompt/change and call "
            'iterate_step(prompt="<their prompt>") — it writes the prompt into the target '
            "node, feeds the chosen image into the wired LoadImage node, runs THIS graph "
            "once, stages the result, and returns a numbered generation history. To revisit "
            'an earlier result, pass from_generation ("original", or a generation number) — '
            'that is how you honour "go back to the original / to generation N, then apply …". '
            "After each run, show the result and ASK the user for the next prompt (or a "
            "go-back); keep looping until they say stop. Do NOT call apply_canvas_hooks, "
            "signal_workflow_ready, or run_research for these."
        )
        for h in iterate_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            tgt = _target_context(h)
            ctx = _input_context(h, base_prompt, hook_id_set)
            prompt_where = (f"prompt → {tgt}" if tgt else
                            "prompt target UNWIRED — ask the user to wire this hook's OUTPUT "
                            "into the prompt node's text input")
            fb_where = (f"feedback image ← {ctx}" if ctx and ctx != "no input wired" else
                        "feedback node UNWIRED — ask the user to wire the LoadImage node's "
                        "image output into this hook's anchor")
            tail = f' — "{directive}"' if directive else ""
            lines.append(f"- ITERATE hook {hid}: {prompt_where}; {fb_where}{tail}")

    if text_hooks:
        lines.append(
            "\nTEXT hooks — each produces a single WRITTEN string (not media). WRITE the "
            "answer yourself (activate a relevant writing skill if it helps). Do NOT "
            "generate images/video, do NOT call apply_canvas_hooks, and do NOT build or "
            "run a workflow. Use the wired context as the SUBJECT of the answer. When the "
            'answer is ready, call place_canvas_text(hook_node_id="<id>", text="<answer>") '
            "ONCE per hook — it delivers the string to the input the hook's output feeds "
            "(shown as 'feeds …'; injected at run time if the hook is kept live, or baked in "
            "if frozen — the hook's own setting) and drops an 'agentY text' node. The answer "
            "also streams into the chat:"
        )
        for h in text_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            ctx = _input_context(h, base_prompt, hook_id_set)
            tgt = _target_context(h)
            where = f" feeds {tgt}" if tgt else " (output unwired — answer streams to chat only)"
            lines.append(f'- TEXT hook {hid} (context: {ctx}){where} — write & place → "{directive}"')

    if standin_hooks:
        chains = _order_standin_chains(standin_hooks)
        singles = [c[0] for c in chains if len(c) == 1]
        multis = [c for c in chains if len(c) > 1]

        def _input_desc(h: dict) -> str:
            anchors = _all_anchor_inputs(h, base_prompt)
            if not anchors:
                return "no input wired — treat the prompt as text-to-media"
            parts = [_render_anchor(aid, atype, inputs) for aid, atype, inputs in anchors]
            if len(parts) == 1:
                return f"input from {parts[0]}"
            return "inputs from " + "; ".join(parts)

        def _output_desc(h: dict) -> str:
            n = _export_count(h)
            if n <= 1:
                return ""
            return (f"; EXPORTS {n} outputs — wire the workflow's {n} results to this "
                    "hook's outputs (any type: image/video/string/int/float)")

        def _chain_wiring(h: dict, prev_id) -> str:
            """Slot map for the links feeding *h* from stage *prev_id* (out→in)."""
            pl = [l for l in (h.get("prev_links") or [])
                  if isinstance(l, dict) and str(l.get("from_hook_id")) == str(prev_id)]
            if not pl:
                return ""
            parts = [f"{l.get('to_input', 'anchor')} ← out{l.get('from_output_slot', 0)}" for l in pl]
            return "  [" + ", ".join(parts) + "]"

        if singles:
            lines.append(
                "\nMAKE-WORKFLOW hooks — each is a self-contained generation request. "
                "For each one, GENERATE a ComfyUI workflow that fulfils the prompt (or, "
                "when a workflow doesn't fit, a Python script written into the scripts "
                "dir from get_agent_output_dirs()), then run it via the normal "
                "generation contract — signal_workflow_ready for a workflow, or run the "
                "script — and let the outputs stage onto the canvas as loader nodes. Do "
                "NOT call apply_canvas_hooks for these. If an anchor is wired, its output "
                "is the INPUT to what you generate (e.g. upload that image/video, or feed "
                "a wired string/number, and bind it to the workflow); if nothing is wired, "
                "treat the prompt as a text-to-media request. A single make_workflow hook may PRODUCE "
                "SEVERAL results (e.g. 'make 4 angle images') and even run more than one "
                "workflow — that's fine; every produced file/value is captured. Media "
                "routing (agent/images, agent/videos, …) is enforced automatically:"
            )
            for h in singles:
                prompt = str(h.get("directive", "") or "").strip()
                lines.append(f'- MAKE-WORKFLOW, {_input_desc(h)}{_output_desc(h)} — generate & run → "{prompt}"')

        if multis:
            lines.append(
                "\nMAKE-WORKFLOW CHAINS — hooks wired output→input, run STRICTLY IN "
                "ORDER. Run each stage with run_workflow_now(workflow_path) (NOT "
                "signal_workflow_ready) so you capture its output(s) to build the next "
                "stage.\n"
                "CRITICAL — every stage AFTER stage 1 RECEIVES the previous stage's "
                "output as its INPUT; it is NEVER a fresh text-to-media request. For "
                "stages 2+ do NOT call run_research and do NOT build a text-to-image "
                "workflow: take the incoming file(s) as the input image/video and build "
                "an image-to-image / edit / image-to-video (or scalar-consuming) workflow "
                "that binds them — upload_image the produced media and wire it into the "
                "loader, or inject a computed scalar as a widget/prompt value. The "
                "per-stage slot map shows which output feeds which input. A stage may "
                "export SEVERAL outputs of any type: a workflow output FILE "
                "(image/video/audio) OR a VALUE you compute from the run (e.g. 'generate "
                "a video AND calculate its length' → the video is one output, the length "
                "another you derive with a tool/script). run_workflow_now returns EVERY "
                "produced file — if a stage yields multiple files (e.g. 'make 4 images'), "
                "forward ALL of them to the next stage (upload_image each and wire them "
                "into the next workflow's loader(s) or a batch), not just the first. "
                "Stage 1's input is its wired "
                "anchor if any, else text-to-media; the final stage's output is the "
                "result. Do NOT call apply_canvas_hooks for these:"
            )
            for ci, chain in enumerate(multis, 1):
                lines.append(f"  Chain {ci}:")
                prev = None
                for si, h in enumerate(chain, 1):
                    prompt = str(h.get("directive", "") or "").strip()
                    if prev is None:
                        in_tag = f"input = {_input_desc(h)}"
                    else:
                        wiring = _chain_wiring(h, prev.get("hook_node_id"))
                        in_tag = ("INPUT = stage %d's output — image-to-media/EDIT, NOT "
                                  "text-to-image%s" % (si - 1, wiring))
                    lines.append(f'    Stage {si} [{in_tag}]: "{prompt}"{_output_desc(h)}')
                    prev = h

        bake_hooks = [h for h in standin_hooks if _wants_bake(h)]
        if bake_hooks:
            lines.append(
                "\nBAKE TO CANVAS — one or more make_workflow hooks above has 'bake_to_canvas' "
                "ON. After you have GENERATED and validated each such stage's workflow, do "
                "NOT stop at running it: call bake_hooks_to_canvas(stages=[…]) to nest each "
                "generated workflow into a ComfyUI subgraph whose inputs/outputs MATCH that "
                "hook's slots, place the subgraphs on the same canvas, and wire them to "
                "mirror the hook chain — baking the multi-step task into a reusable native "
                "workflow the user can re-run without you. For each baked stage pass: "
                "workflow_path (the generated workflow), hook_node_id, exposed inputs "
                "(a node_id+input_name per anchor, in slot order — where the wired input "
                "binds inside the workflow), exposed outputs (a node_id+output_slot per "
                "exported result, any type), and prev_hook_ids (its predecessor stage[s]). "
                "For a value you computed OUTSIDE the graph at runtime (e.g. a measured "
                "length via run_script), pass it in the stage's `computed_outputs`: the "
                "SAME Python snippet, plus which inner outputs feed it (node_id + "
                "output_slot, bound as in0, in1, …). Bake injects an AgentYPython node "
                "running that snippet and exposes its result, so the value is reproduced "
                "on re-run without you. Stages NOT marked bake are generated/run as usual, "
                "not baked. The baked subgraphs are ADDED to the canvas next to the hook "
                "nodes — the hooks are NOT removed."
            )
            for h in bake_hooks:
                hid = h.get("hook_node_id")
                lines.append(
                    f'  - bake hook {hid}: "{str(h.get("directive", "") or "").strip()}" '
                    f'({_input_desc(h)}; exports {_export_count(h)} output(s))'
                )

    lines.append("")
    return "\n".join(lines)
