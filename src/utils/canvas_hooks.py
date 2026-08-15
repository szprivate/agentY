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
import os
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


def _target_input_types(hooks) -> dict:
    """``(hook_id, input_name) -> declared wire type`` from the frontend's targets."""
    out: dict = {}
    for h in (hooks or []):
        if not isinstance(h, dict) or h.get("hook_node_id") is None:
            continue
        hid = str(h["hook_node_id"])
        for t in (h.get("targets") or []):
            if isinstance(t, dict) and t.get("to_input"):
                out[(hid, str(t["to_input"]))] = str(t.get("to_input_type") or "")
    return out


def _anchor_out_types(hooks) -> dict:
    """``hook_id -> {anchor_node_id: that anchor's output wire type}``."""
    out: dict = {}
    for h in (hooks or []):
        if not isinstance(h, dict) or h.get("hook_node_id") is None:
            continue
        types = {str(a["node_id"]): str(a.get("from_output_type") or "")
                 for a in (h.get("anchors") or [])
                 if isinstance(a, dict) and a.get("node_id") is not None}
        out[str(h["hook_node_id"])] = types
    return out


# Slot types that carry no promise about what flows through them: a reroute, an
# agentY ref note, an unwired-yet MatchType. They pass whatever they were given,
# so they are compatible with any target rather than with none.
_WILDCARD_TYPES = {"", "*", "COMFY_MATCHTYPE_V3", "COMFY_MULTITYPE_V3", "ANY"}


def _type_fits(anchor_type: str, wire_type: str) -> bool:
    """Whether an anchor's output can feed an input declared *wire_type*."""
    a = str(anchor_type or "").strip()
    w = str(wire_type or "").strip()
    if not w or w in _WILDCARD_TYPES or a in _WILDCARD_TYPES:
        return True
    return a == w or w in {t.strip() for t in a.split(",")}


def splice_hook_nodes(prompt: dict, hooks: list | None = None) -> tuple[dict, list[str]]:
    """Return ``(clean_prompt, removed_ids)``.

    Removes every ``AgentYHook`` node from an API-format *prompt*. For an inline
    hook (its output feeds a downstream node) each downstream input is rewired to
    the hook's own ``anchor`` source so the graph stays connected. The anchor
    input auto-grows (``anchor``, ``anchor0``, ``anchor1``, …), so a hook may have
    several wired inputs; with *hooks* supplied the anchor whose output type
    matches the target input is preferred, else the first (lowest-slot) is used,
    matching the node's passthrough. A dangling hook (nothing consumes it) is
    simply dropped. Works whether or not ComfyUI's ``graphToPrompt`` already pruned
    the hook.

    A hook wired into a **widget-backed** input (``STRING``, ``INT``, …) is
    producing that input's *value*, so there is nothing to pass through: the link
    is dropped and the input left for the produced value to fill. Passing the
    anchor through regardless is how a ``LoadImage`` ended up wired into a prompt
    box — inert on a normal run, and impossible for the agent to overwrite once it
    looked like a connection. Needs *hooks* for the declared types; without them
    every consumer is rewired as before.
    """
    if not isinstance(prompt, dict):
        return prompt, []
    clean = copy.deepcopy(prompt)
    hook_ids = [nid for nid, node in clean.items()
                if isinstance(node, dict) and node.get("class_type") == _HOOK_CLASS]
    if not hook_ids:
        return clean, []
    target_types = _target_input_types(hooks)
    anchor_types = _anchor_out_types(hooks)
    # An anchor drawn through an `agentY ref note` arrives at the hook FROM the
    # note, so the graph's link names the note while the hook payload reports the
    # node it wraps. Without this the anchor's type is unknown at exactly the
    # moment it decides which wire replaces the hook.
    _roles, wrapped = ref_notes(clean)
    for hid in hook_ids:
        node = clean.get(hid, {}) or {}
        links = _anchor_links(node.get("inputs") or {})
        by_node = anchor_types.get(str(hid), {})

        def _anchor_type(nid, _by=by_node) -> str:
            nid = str(nid)
            return _by.get(nid) or _by.get(str(wrapped.get(nid, ""))) or ""

        def _source_for(wire_type: str):
            """The anchor to pass through for a target of *wire_type*, or None.

            Exact type first, then anything compatible (a reroute or a ref note
            declares a wildcard and carries whatever it was handed). If the target
            has a declared type and we know the anchors' types, and none of them
            fits, the answer is **None** — leaving the input unwired. Passing the
            first anchor through regardless is how a prompt string was wired into
            an image input: the graph then fails validation at submission, which
            reads as an executor bug rather than as a hook with nothing suitable
            on it. Only when no anchor type is known at all (an older frontend
            that sends none) does the original first-link behaviour apply.
            """
            if not links:
                return None
            if wire_type:
                for link in links:
                    if _anchor_type(link[0]) == wire_type:
                        return link
                if any(_anchor_type(link[0]) for link in links):
                    return next((link for link in links
                                 if _type_fits(_anchor_type(link[0]), wire_type)), None)
            return links[0]

        # Rewire any consumer of this hook's output back to the hook's source.
        for other in clean.values():
            if not isinstance(other, dict):
                continue
            for k, v in list((other.get("inputs") or {}).items()):
                if not (isinstance(v, list) and len(v) == 2 and str(v[0]) == str(hid)):
                    continue
                declared = target_types.get((str(hid), str(k)))
                # Unknown type (no hook metadata) → pass through, as before.
                passthrough = not declared or is_connection_type(declared)
                src = _source_for(declared or "") if passthrough else None
                if src is not None:
                    other["inputs"][k] = list(src)
                else:
                    other["inputs"].pop(k, None)
        clean.pop(hid, None)
    return clean, hook_ids


def hook_scoped_graph() -> bool:
    """Whether a hook run is trimmed to the branch(es) its hooks reach.

    ``AGENTY_HOOK_SCOPE`` wins when set; otherwise ``hook_scoped_graph`` in
    settings (default on). Off restores the old behaviour — the whole canvas runs,
    every unrelated output branch included.
    """
    env = os.environ.get("AGENTY_HOOK_SCOPE")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("hook_scoped_graph", True))
    except Exception:  # noqa: BLE001 — never let settings break a turn
        return True


def _ancestors(prompt: dict, roots) -> set:
    """*roots* plus every node they transitively depend on (their input closure)."""
    seen: set = set()
    stack = [str(r) for r in roots]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        for value in ((prompt[nid] or {}).get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2:
                stack.append(str(value[0]))
    return seen


def _descendants(prompt: dict, roots) -> set:
    """*roots* plus every node that transitively consumes them (output closure)."""
    children: dict = {}
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        for value in ((node.get("inputs") or {}).values()):
            if isinstance(value, list) and len(value) == 2:
                children.setdefault(str(value[0]), set()).add(str(nid))
    seen: set = set()
    stack = [str(r) for r in roots]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        stack.extend(children.get(nid, ()))
    return seen


def hook_scope_ids(prompt: dict, hook_ids=None) -> set | None:
    """Ids of the part of *prompt* the executed hooks actually reach.

    ``None`` means "no hooks to scope to" — the caller keeps the whole graph.

    Seeded from each hook AND from the node(s) its anchors read: a hook whose own
    output is unwired (an ``inline_parameter`` sweeping a widget on its anchor)
    still governs everything downstream of that anchor, and seeding only from the
    hook would prune the very branch it mutates.

    From those seeds the scope is ``ancestors(descendants(seeds))``. The
    descendants are what the hook affects; taking the ancestors of *those* is what
    keeps the result runnable — a kept KSampler needs its model/latent/conditioning
    even though none of that sits downstream of the hook. Sibling branches that
    merely share an upstream loader come along as ancestors only if something kept
    actually consumes them, so unrelated output chains drop out.
    """
    hooks_in_prompt = {str(nid) for nid, node in prompt.items()
                       if isinstance(node, dict) and node.get("class_type") == _HOOK_CLASS}
    if not hooks_in_prompt:
        return None
    seeds = set(hooks_in_prompt)
    if hook_ids:
        # Honour "the hooks being executed": a bypassed/muted hook is not collected
        # by the frontend, so it must not drag its branch into the run either.
        chosen = {str(h) for h in hook_ids if h is not None} & hooks_in_prompt
        if chosen:
            seeds = chosen
    for hid in list(seeds):
        for link in _anchor_links((prompt.get(hid) or {}).get("inputs") or {}):
            seeds.add(str(link[0]))
    if not seeds:
        return None
    return _ancestors(prompt, _descendants(prompt, seeds))


def prune_to_hooks(prompt: dict, hook_ids=None) -> tuple[dict, list]:
    """Return ``(scoped_prompt, dropped_ids)``.

    Trims an API-format *prompt* to :func:`hook_scope_ids`. Without this a hook on
    one branch of a large canvas runs — and gets written into every generated
    workflow — together with every unrelated branch on the graph, because ComfyUI
    executes each output node it is given. Returns the prompt untouched when there
    is nothing to scope to or nothing to drop.
    """
    if not isinstance(prompt, dict) or not prompt:
        return prompt, []
    keep = hook_scope_ids(prompt, hook_ids)
    if not keep:
        return prompt, []
    dropped = [str(nid) for nid in prompt if str(nid) not in keep]
    if not dropped:
        return prompt, []
    return {nid: node for nid, node in prompt.items() if str(nid) in keep}, dropped


# Wire types a widget can carry as a literal. Everything else (IMAGE, LATENT,
# MODEL, MASK, AUDIO, …) only ever travels down a CONNECTION, so a produced value
# for such an input has to become a link to a node that makes one — writing the
# value in place replaces the wire and silently disconnects the input.
_PRIMITIVE_WIRE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL", "NUMBER", "COMBO"}

# Widget names that name a file on the loader nodes we may reuse or clone.
_FILE_WIDGETS = ("image", "video", "file", "filename", "audio")

_AUDIO_EXTS = {"mp3", "wav", "flac", "ogg", "m4a"}
_MEDIA_EXTS = IMG_EXTS | VID_EXTS | _AUDIO_EXTS


def _looks_like_media_file(value: str) -> bool:
    """Whether *value* names a media file, rather than being prose or an id.

    Guards the clone/create paths: without it any produced string would be turned
    into a loader pointing at a file that does not exist, which fails at run time
    instead of being reported as unresolvable up front.
    """
    parts = _basename(value).rsplit(".", 1)
    return len(parts) == 2 and parts[1].lower() in _MEDIA_EXTS


def is_connection_type(wire_type: str | None) -> bool:
    """Whether a wire of *wire_type* must be a link rather than a literal value."""
    return bool(wire_type) and str(wire_type).strip().upper() not in _PRIMITIVE_WIRE_TYPES


def _basename(value) -> str:
    return Path(str(value).replace("\\", "/")).name


def _free_id(prompt: dict) -> str:
    nums = [int(k) for k in prompt if str(k).isdigit()]
    return str((max(nums) if nums else 0) + 1)


def _node_loading(prompt: dict, filename: str) -> str | None:
    """Id of a node already loading *filename* (matched on basename)."""
    want = _basename(filename)
    if not want:
        return None
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key in _FILE_WIDGETS:
            cur = inputs.get(key)
            if isinstance(cur, str) and cur and _basename(cur) == want:
                return str(nid)
    return None


def as_connection(prompt: dict, value, current=None) -> list | None:
    """Resolve a produced *value* to a ``[node_id, slot]`` for a connection input.

    *current* is what the input holds now (the link the hook was spliced onto),
    which is the best clue to what kind of source the target expects.

    Accepts, in order: an explicit ``[node_id, slot]``; the id of a node already on
    the canvas (the natural answer when the user wired several images into the hook
    and one of them is to be selected); or a filename — reusing whatever node
    already loads that file, else **cloning** the node currently feeding the input
    and pointing the clone at the new file, else adding a ``LoadImage``. Cloning
    keeps the user's own loader class (and leaves the original alone for whatever
    else consumes it).

    ``None`` means "cannot be expressed as a connection" — the caller must skip
    rather than write a literal, which would disconnect the input.
    """
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [str(value[0]), int(value[1] or 0)]
        except (TypeError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    if text in prompt:                      # the agent named a node to connect
        return [text, 0]
    hit = _node_loading(prompt, text)       # something already loads this file
    if hit:
        return [hit, 0]
    if not _looks_like_media_file(text):
        return None                         # prose / an unknown id — not wireable
    src = None
    if isinstance(current, (list, tuple)) and len(current) == 2:
        src = prompt.get(str(current[0]))
    if isinstance(src, dict):               # clone the user's own loader
        for key in _FILE_WIDGETS:
            if isinstance((src.get("inputs") or {}).get(key), str):
                nid = _free_id(prompt)
                clone = copy.deepcopy(src)
                clone.setdefault("inputs", {})[key] = text
                prompt[nid] = clone
                return [nid, 0]
    if _basename(text).rsplit(".", 1)[-1].lower() in IMG_EXTS:
        nid = _free_id(prompt)
        prompt[nid] = {"class_type": "LoadImage",
                       "inputs": {"image": text, "upload": "image"}}
        return [nid, 0]
    return None


def _write_input(prompt: dict, node: dict, param: str, value,
                 connection: bool = False) -> bool:
    """Set *param* on *node*, as a link when the input is a connection.

    Returns False when a connection input could not be resolved — the input is
    then left as it was rather than being overwritten with a literal.

    *connection* says the input takes a wire even though the graph does not
    currently show one. That happens as a matter of course now: splicing removes
    the hook that fed it and leaves it unwired when no anchor could replace it, so
    "is there a link here" stopped being the same question as "does this take a
    link". Getting it wrong writes a string into an IMAGE input, which ComfyUI
    accepts and the node reports at run time as
    ``'str' object has no attribute 'shape'``.
    """
    inputs = node.setdefault("inputs", {})
    current = inputs.get(param)
    wired = isinstance(current, list) and len(current) == 2
    empty = value is None or (isinstance(value, str) and not value.strip())
    if wired or connection:
        # An empty value for a connection means "nothing on this one": leave it
        # unwired. "Use a reference where I have one, otherwise leave the image
        # inputs empty" is an ordinary directive, and without this there is no way
        # to express it. Only valid for an optional input; a required one then
        # fails validation, which is the truth.
        if empty:
            inputs.pop(param, None)
            return True
        link = as_connection(prompt, value, current if wired else None)
        if link is None:
            return False
        inputs[param] = link
        return True
    inputs[param] = value
    return True


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


def build_batch(base_prompt: dict, resolutions: list, cap: int = 25,
                labels: list | None = None,
                connection_inputs: set | None = None) -> tuple[list[dict], list[str]]:
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

    Pass a list as *labels* and it is filled with one ``{"<node>.<param>": value}``
    dict per prompt, in the same order — what each variant was actually made from.
    Without it a batch of five is five anonymous graphs, and pairing "the third
    one" with "the reference frame for Ben" is left to whoever is counting.

    *connection_inputs* (``{"<node>.<input>"}``, from
    :func:`connection_targets`) names the inputs that take a WIRE even when the
    graph shows none — an input whose hook was spliced out and left unwired still
    needs a link, and a literal written there reaches the node as a string.
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
    unresolved: dict = {}
    for combo in combos:
        p = copy.deepcopy(base_prompt)
        label: dict = {}
        for row in combo:                       # each row is one group's aligned assignments
            for (nid, param, val) in row:
                label[f"{nid}.{param}"] = val
                node = p.get(nid)
                if not isinstance(node, dict):
                    continue
                if not _write_input(p, node, param, val,
                                    f"{nid}.{param}" in (connection_inputs or ())):
                    # A connection input (IMAGE, LATENT, …) we could not turn into
                    # a link. Leave the wire intact and say so once, rather than
                    # writing a literal that would disconnect it.
                    unresolved.setdefault(f"{nid}.{param}", set()).add(str(val)[:60])
        prompts.append(p)
        if labels is not None:
            labels.append(label)
    for slot, vals in unresolved.items():
        notes.append(
            f"{slot} is a connection input — could not wire "
            + ", ".join(sorted(vals))
            + " to a node that produces it; left the existing wire in place. Give a "
              "node id to connect (e.g. one of the hook's anchors), or a file the "
              "canvas can load.")
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


_QA_PURPOSES = {"qa", "quality", "check", "review", "qa_check", "qa-check"}


def _is_qa(hook: dict) -> bool:
    """True if *hook* carries a QA briefing rather than asking for work.

    A qa hook produces nothing and is never "run": its directive is the checklist
    and its anchors are reference/mood images. It is read by
    :mod:`src.utils.qa` after a generation finishes, to judge what came out.
    """
    return str(hook.get("purpose", "") or "").strip().lower() in _QA_PURPOSES


_GENERAL_PURPOSES = {"general_request", "general-request", "general", "request",
                     "free", "freeform", "free_form", "free-form"}


def _is_general(hook: dict) -> bool:
    """True if *hook* is a free-form ``general_request``: the agent interprets the
    directive as an ordinary request (wired anchors = provided inputs/context,
    graph already captured) and picks the action itself, rather than the specific
    producer / text / make_workflow / iterate mechanics."""
    return str(hook.get("purpose", "") or "").strip().lower() in _GENERAL_PURPOSES


def _wants_bake(hook: dict) -> bool:
    """True if *hook* has the ``bake`` switch on (bake to a subgraph).

    On a make_workflow hook ``bake`` means "nest the generated workflow into a
    subgraph"; on the place_canvas_text purposes the same switch means "bake the
    value into the target input" and is read as ``freeze`` there. One question on
    the node, resolved by purpose — see the node's docstring.
    """
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


_REF_NOTE_CLASS = "AgentYRefNote"
_REF_NOTE_HOPS = 4  # notes on notes: follow a few, then stop rather than loop


def ref_notes(base_prompt: dict | None) -> tuple[dict, dict]:
    """Read the ``agentY ref note`` nodes off the graph.

    A ref note sits ON the wire that carries a reference — LoadImage → ref note →
    wherever — and says what the agent should take from it ("the face, not the
    styling"). Living on the wire is the point: there is no node id to keep in
    sync, because whatever is plugged into the note is what the note is about.

    Returns ``(role_by_node_id, wrapped_by_note_id)``. The first answers "does this
    input come with a stated role", keyed by the node the user actually recognises
    (the loader) *and* by the note itself. The second lets an anchor drawn on the
    note be reported as the node behind it — "node 51 (AgentYRefNote)" tells the
    agent nothing about what it is looking at.
    """
    roles: dict = {}
    wrapped: dict = {}
    if not isinstance(base_prompt, dict):
        return roles, wrapped

    notes = {nid: node for nid, node in base_prompt.items()
             if isinstance(node, dict) and node.get("class_type") == _REF_NOTE_CLASS}

    def _source(nid: str) -> str | None:
        link = ((notes[nid].get("inputs") or {}).get("input"))
        return str(link[0]) if isinstance(link, list) and link else None

    for nid, node in notes.items():
        role = str((node.get("inputs") or {}).get("role") or "").strip()
        # Walk back to the first node that isn't itself a note, so a note stacked
        # on a note still names the loader rather than the note below it.
        src, hops = _source(nid), 0
        while src in notes and hops < _REF_NOTE_HOPS:
            src, hops = _source(src), hops + 1
        if src is not None:
            wrapped[nid] = src
        if not role:
            continue
        roles[nid] = role
        if src is not None and src not in roles:
            roles[src] = role
    return roles, wrapped


def _all_anchor_inputs(hook: dict, base_prompt: dict | None) -> list:
    """Return ``[(anchor_id, anchor_type, scalar_inputs_dict, tap, role), …]`` for
    every real-node input wired to a hook.

    The anchor input auto-grows, so a hook may gather several inputs (carried in
    the ``anchors`` list). Falls back to the singular ``anchor_node_id`` field for
    older frontends that only send one. *tap* is ``(wire_type, paths)`` when
    :mod:`src.utils.canvas_tap` rendered this anchor's wire to disk — it carried a
    runtime tensor rather than a named file — and ``None`` for everything else.
    *role* is what an ``agentY ref note`` on this input says the reference is FOR,
    or ``""`` — an anchor drawn on the note itself is reported as the node the note
    wraps, since the note is an annotation on the wire, not the subject.
    """
    entries: list = []
    plural = hook.get("anchors")
    if isinstance(plural, list) and plural:
        for a in plural:
            if isinstance(a, dict) and a.get("node_id") is not None:
                entries.append((str(a["node_id"]), a.get("type"), a.get("widgets"),
                                a.get("tapped_type"), a.get("tapped"),
                                str(a.get("role") or "").strip(),
                                str(a.get("title") or "").strip(),
                                str(a.get("to_input") or "")))
    elif hook.get("anchor_node_id") is not None:
        entries.append((str(hook["anchor_node_id"]), hook.get("anchor_type"),
                        hook.get("anchor_widgets"), None, None, "",
                        str(hook.get("anchor_title") or "").strip(), ""))

    roles, wrapped = ref_notes(base_prompt)
    out: list = []
    seen: set = set()
    for aid, atype, widgets, wire, tapped, sent, title, slot in entries:
        # The frontend already resolves a note on the anchor's own wire (so every
        # consumer sees the real node); reading the graph catches the rest — a note
        # elsewhere on that loader, or an older frontend that sends neither.
        role = sent or roles.get(aid, "")
        if aid in wrapped:
            # The anchor is the note; the subject is what it wraps.
            aid = wrapped[aid]
            atype = None
            widgets = None
            title = ""
        if aid in seen:
            continue
        seen.add(aid)
        inputs: dict = {}
        if base_prompt and aid in base_prompt:
            raw = (base_prompt[aid].get("inputs") or {})
            inputs = {k: v for k, v in raw.items() if not isinstance(v, list)}
            atype = atype or str(base_prompt[aid].get("class_type") or "")
            title = title or str((base_prompt[aid].get("_meta") or {}).get("title") or "")
        elif isinstance(widgets, dict):
            inputs = widgets
        paths = [str(p) for p in (tapped or []) if str(p).strip()]
        out.append((aid, atype or "?", inputs,
                    (str(wire or "live"), paths) if paths else None,
                    role or roles.get(aid, ""), title, slot))
    return out


def _slot_label(to_input: str) -> str:
    """``anchors.anchor1`` → ``anchor_1``: the name the user writes in a directive.

    Directives say "the prompts in anchor_0, the references in anchor_1" all the
    time. Listing the inputs without saying which slot each arrived on leaves the
    agent to guess that mapping from order — and with five references and two
    chained hooks feeding one node, guessing is exactly what goes wrong.
    """
    m = re.search(r"anchor[_\-]?(\d*)$", str(to_input or "").strip(), re.I)
    if not m:
        return ""
    return f"anchor_{m.group(1) or '0'}"


def _slot_order(to_input: str) -> int:
    """Sort key for an anchor slot; unknown slots keep their arrival order."""
    m = re.search(r"anchor[_\-]?(\d+)$", str(to_input or "").strip(), re.I)
    return int(m.group(1)) if m else 10_000


def _chain_inputs(hook: dict, hook_ids: set) -> list:
    """``[(slot_name, producing_hook_id), …]`` — the hooks wired into this one.

    The frontend files hook→hook links under ``prev_links`` rather than with the
    real-node anchors, so a hook fed ONLY by other hooks used to render as "no
    input wired" — while its own directive talked about anchor_0 and anchor_1.
    """
    out: list = []
    seen: set = set()
    for link in (hook.get("prev_links") or []):
        if not isinstance(link, dict):
            continue
        hid = str(link.get("from_hook_id") or "")
        if not hid or hid in seen:
            continue
        seen.add(hid)
        out.append((str(link.get("to_input") or ""), hid))
    for hid in (hook.get("prev_hook_ids") or []):
        hid = str(hid)
        if hid and hid not in seen and hid in hook_ids:
            seen.add(hid)
            out.append(("", hid))
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
    for tid, _ttype, tin, tintype, _ttitle in _output_targets(hook):
        node = base_prompt.get(tid)
        if not isinstance(node, dict) or not tin:
            continue
        # A connection input is delivered as a link, never as a literal — writing
        # the value straight in would replace the wire and disconnect the input.
        # The hook's own metadata says which is which; the graph no longer does,
        # since splicing may already have removed the wire this replaces.
        if _write_input(base_prompt, node, tin, value, is_connection_type(tintype)):
            written.append(tid)
    return written


def missing_collector_files(prompt: dict | None) -> list:
    """Lines in a collector's ``files`` that name nothing on disk.

    The collector keeps the files it can find and **silently skips the rest** —
    which is the right behaviour for a list a human curated, and a trap for a list
    an agent wrote. Every dropped line shifts the numbering of everything after
    it, and the numbering is the whole contract when a prompt says
    ``@image3 walks past @image4``: the video comes back with the wrong characters
    in it, and nothing anywhere reported an error.
    """
    out: list = []
    for nid, node in (prompt or {}).items():
        if not isinstance(node, dict) or node.get("class_type") not in _COLLECTOR_TYPES:
            continue
        val = (node.get("inputs") or {}).get("files")
        if not isinstance(val, str) or not val.strip():
            continue
        lines = [ln.strip().strip('"') for ln in val.splitlines() if ln.strip()]
        missing = [ln for ln in lines if not Path(ln).is_file()]
        if missing:
            out.append({"node_id": str(nid), "class_type": str(node.get("class_type")),
                        "lines": len(lines), "missing": missing[:12]})
    return out


def connection_targets(hooks: list | None) -> set:
    """``{"<node>.<input>"}`` for every hook target that takes a WIRE, not a value.

    Read from the declared type the frontend recorded for each target, which is
    the only place that still knows: by the time a batch is built, the hook has
    been spliced out and an input it fed may hold no link at all.
    """
    out: set = set()
    for h in (hooks or []):
        if not isinstance(h, dict):
            continue
        for tid, _ttype, tin, tintype, _ttitle in _output_targets(h):
            if tin and is_connection_type(tintype):
                out.add(f"{tid}.{tin}")
    return out


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


# A directive that gates the run on how an EARLIER step turned out — "if ANY
# reference generation failed, STOP", "wait for all the shots", "only continue
# once …". These need results, and results only exist for work that was RUN, so
# whatever such a hook depends on must not be left queued for the end of the turn.
_OUTCOME = (r"fail(?:ed|s|ure|ures)?|error(?:s|ed)?|missing|empty|succe(?:ed|eded|ss|ssful)|"
            r"complete[d]?|finish(?:ed)?|generated|exists?|worked|came out")
_CONDITIONAL_PATTERNS = (
    # Hyphen-aware boundaries: "a non-stop dolly move" is camera work, not a halt.
    re.compile(r"(?<![\w-])(?:stop|abort|halt)(?![\w-])", re.I),
    re.compile(r"\bdo\s*n[o']?t\s+(?:continue|proceed|go on)\b", re.I),
    re.compile(r"\bwait\s+(?:for|until|till)\b", re.I),
    re.compile(r"\bonly\s+(?:if|when|once|after)\b", re.I),
    # "only continue once every shot exists" — the verb sits between the two halves.
    re.compile(r"\bonly\s+(?:continue|proceed|go\s+on|start|run|generate|build|do)\b", re.I),
    re.compile(rf"\bif\s+(?:any|all|none|one|the|it|they|there)\b[^.!?]{{0,80}}\b(?:{_OUTCOME})\b", re.I),
)


def is_conditional(hook: dict) -> bool:
    """True when the hook's directive makes continuing depend on an earlier outcome."""
    text = str((hook or {}).get("directive", "") or "")
    return any(p.search(text) for p in _CONDITIONAL_PATTERNS)


# What the user said this hook's OUTPUTS are, stated in the hook's own prompt.
# Explicit syntax first ("role: hero sheet", "[role: shot start frame]"), then
# the way people write it in a sentence ("tag the outputs as 'alley night'").
# Deliberately not inferred from the directive as a whole: a role that fires on
# every hook is a label nobody trusts, and it is used to DECORATE the user's
# canvas — an auto ref note per output — which has to be something they asked for.
_ROLE_PATTERNS = (
    re.compile(r"\[\s*roles?\s*[:=]\s*([^\]\n]+)\]", re.I),
    re.compile(r"^[\s\-*>]*roles?\s*[:=]\s*(.+?)\s*$", re.I | re.M),
    re.compile(r"\b(?:tag|label|name|mark)\s+(?:the\s+)?"
               r"(?:outputs?|results?|images?|videos?|frames?|them|these|it)\s+"
               r"(?:as|with)\s+[\"“'‘]?([^\"”'’.\n]+)", re.I),
)
_MAX_ROLE = 80


def declared_output_role(hook_or_text) -> str:
    """The role the user stated for this hook's outputs, or ''.

    Accepts a hook dict or a raw directive. The value is what a generated file
    gets tagged with — its sidecar, the title of the node dropped for it, and
    (only when it is stated like this) an ``agentY ref note`` attached to that
    node, so the next run reads the user's own words rather than a filename.
    """
    text = hook_or_text if isinstance(hook_or_text, str) else \
        str((hook_or_text or {}).get("directive", "") or "")
    for pat in _ROLE_PATTERNS:
        m = pat.search(text)
        if m:
            role = " ".join(m.group(1).split()).strip(" -–—:;,\"'")
            if role:
                return role[:_MAX_ROLE]
    return ""


def _producers_of(hooks: list) -> dict:
    """consumer hook id -> {producer hook ids}, from BOTH ends of the wire.

    A hook→hook link is recorded on whichever side the frontend saw it: as an
    anchor on the consumer (``prev_hook_ids`` / ``anchors[].node_id``), or as a
    target on the producer (hook 5 *feeds* hook 30's ``anchors.anchor0``). Real
    graphs use the target side — a consumer whose anchors are fed only by hooks
    reports "no input wired" — so reading one end alone finds no dependencies at
    all and the plan silently degrades to nothing.
    """
    ids = _hook_ids(hooks)
    producers: dict = {i: set() for i in ids}
    for h in hooks:
        hid = str(h.get("hook_node_id"))
        for pid in _hook_predecessors(h, ids):        # recorded on the consumer
            producers.setdefault(hid, set()).add(pid)
        for tid, _ty, _ti, _tt, _title in _output_targets(h):   # on the producer
            if tid in ids and tid != hid:
                producers.setdefault(tid, set()).add(hid)
    return producers


def _is_chain_only(hook: dict, hook_ids: set) -> bool:
    """The hook's output is wired, and every input it reaches belongs to a hook.

    Only says yes when targets were actually recorded — a hook with none at all is
    unknown, not chain-only, and keeps whatever behaviour the caller had.
    """
    real, chain = _split_targets(hook, hook_ids)
    return bool(chain) and not real


def gating_hook_ids(hooks: list) -> set:
    """Ids of hooks whose RESULTS a conditional hook depends on.

    Everything upstream of a conditional hook, transitively: those hooks have to
    be *run* (``apply_canvas_hooks(run_now=True)`` / ``run_workflow_now``) rather
    than queued, or there is nothing for the condition to read — the turn ends,
    the batch runs afterwards, and the check never happens.
    """
    producers = _producers_of(hooks)
    gating: set = set()
    for h in hooks:
        if not is_conditional(h):
            continue
        hid = str(h.get("hook_node_id"))
        # Per-hook walk: a cycle (or a hook wired back into its own producer) must
        # not make the conditional hook gate on itself, but a hook upstream of a
        # *different* conditional hook still counts.
        seen: set = set()
        stack = list(producers.get(hid, ()))
        while stack:
            pid = stack.pop()
            if pid in seen or pid == hid:
                continue
            seen.add(pid)
            stack.extend(producers.get(pid, ()))
        gating |= seen
    gating.discard(None)
    # A text hook produces a written string with place_canvas_text — there is no
    # execution to run early, so listing it here would only add noise. A producer
    # whose output only reaches other HOOKS is in the same position: it fills no
    # graph input, so there is nothing to execute and telling the agent to run it
    # sends it to apply_canvas_hooks, which can only answer "no batch was produced".
    ids = _hook_ids(hooks)
    quiet = {str(h.get("hook_node_id")) for h in hooks
             if _is_text(h) or _is_chain_only(h, ids)}
    return gating - quiet


def plan_lines(hooks: list) -> list[str]:
    """The RUN PLAN block: order, what must be run rather than queued, what gates.

    Derived from the wiring and the directives, not from the model — the ordering
    is already a topological sort, and whether a step's *results* are needed in
    this turn is decided by whether a downstream directive is conditional. The
    plan exists because getting this wrong is silent: the agent queues a batch,
    reaches a conditional hook it cannot evaluate, and stops — cancelling the very
    work the condition was about.
    """
    if not hooks:
        return []
    gating = gating_hook_ids(hooks)
    conditional = [h for h in hooks if is_conditional(h)]
    if not conditional:
        return []
    order = " → ".join(str(h.get("hook_node_id")) for h in hooks)
    out = [
        "\nRUN PLAN (derived from the wiring — follow it):\n"
        "  1. Say this plan in the chat before you start on it — one short numbered "
        "line per hook, in your own words. Then get on with it: it is an announcement, "
        "not a question (wait for an answer only if a [PLAN APPROVAL] block says to)."
    ]
    out.append(f"  2. Work the hooks in this order: {order}.")
    if gating:
        ids = ", ".join(sorted(gating, key=lambda x: (len(x), x)))
        out.append(
            f"  3. Hook(s) {ids} must be RUN THIS TURN, not queued — a later hook's "
            "directive is conditional on how they turn out. When such a hook generates, "
            "call apply_canvas_hooks(..., run_now=true) (or run_workflow_now for a single "
            "workflow): both execute immediately and return per-variant success/failure, "
            "which is what the condition reads. apply_canvas_hooks WITHOUT run_now defers "
            "to the end of the turn — its results do not exist while you are still working, "
            "so a condition over them can never be evaluated."
        )
    producers = _producers_of(hooks)
    for h in conditional:
        hid = h.get("hook_node_id")
        deps = sorted(producers.get(str(hid), ()), key=lambda x: (len(x), x))
        dep_txt = f" (reads hook {', '.join(deps)})" if deps else ""
        out.append(
            f"  4. Hook {hid} is CONDITIONAL{dep_txt}: evaluate its condition against the "
            "results you actually have. If the condition to stop is met, call "
            'stop_hook_run(reason="…", question="…") and reply — do not queue more work. '
            "If it is not met, carry on normally."
        )
    return out


def _project_memory_names() -> list[str]:
    """Entry names the current project has on record, or [] if there is no store.

    Lazy and best-effort by design: the sitrep is a courtesy, and a project with no
    ComfyUI to ask (or no memory yet) must simply produce one fewer line.
    """
    try:
        from src.utils.project_memory import list_entries
        return [e.name for e in list_entries()]
    except Exception:  # noqa: BLE001
        return []


def _mentions(directive: str, name: str) -> bool:
    """Whether *directive* names a stored entry.

    Entry names are slugs ("hero", "alley-night"), so match the words rather than
    the raw slug — nobody writes "alley-night" in a sentence, they write "the alley
    at night". Every word has to be there, and each on a word boundary: substring
    matching would fire on "grade" inside "upgraded", which is exactly the kind of
    wrong that makes a warning block get ignored by the third turn.
    """
    words = [w for w in name.split("-") if w]
    if not words:
        return False
    text = directive or ""
    return all(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def sitrep_lines(hooks: list, base_prompt: dict | None = None,
                 known: list | None = None) -> list[str]:
    """What this turn is about to assume, stated before it acts on it.

    The RUN PLAN says what will happen; this says what is *unresolved* about it.
    Everything here is computed from the wiring, the directives and the project's
    own memory — never from the model — so it costs nothing at run time and cannot
    drift from what the graph actually says.

    Deliberately silent unless something is genuinely open: a block that appears
    every turn stops being read by the third one.
    """
    if not hooks:
        return []
    ids = _hook_ids(hooks)
    producers = _producers_of(hooks)
    names = _project_memory_names() if known is None else list(known)
    items: list[str] = []

    for h in hooks:
        hid = str(h.get("hook_node_id"))
        directive = str(h.get("directive") or "").strip()
        anchors = _all_anchor_inputs(h, base_prompt)
        real, chain = _split_targets(h, ids)

        if not _is_text(h) and _is_chain_only(h, ids):
            who = ", ".join(f"hook {t}" for t in sorted({t[0] for t in chain},
                                                        key=lambda s: (len(s), s)))
            items.append(
                f"hook {hid} feeds only {who} — nothing it produces reaches a node that "
                "renders. Assuming it is a written value, not a generation."
            )
        if not anchors and not _chain_inputs(h, ids) and not real and not chain \
                and not _is_standin(h):
            items.append(
                f"hook {hid} has nothing wired in and nothing wired out. Assuming its "
                "directive stands on its own; wire an anchor if it was meant to act on "
                "something."
            )
        if is_conditional(h) and not producers.get(hid):
            items.append(
                f"hook {hid} waits on how something turns out, but nothing upstream of it "
                "produces a result. Assuming there is nothing to wait for."
            )
        for name in names:
            if _mentions(directive, name):
                items.append(
                    f'hook {hid} mentions "{name}", which this project already has on '
                    f'record — read it with project_memory_read("{name}") and use what is '
                    "stored instead of writing your own version."
                )

    if not items:
        return []
    return ["\nUNRESOLVED — how this turn will be read unless you say otherwise:"] + \
           [f"  • {t}" for t in items]


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


def _recorded_role(inputs: dict) -> str:
    """What a file this node loads was recorded as being FOR, or ''.

    agentY writes a ``.agenty.json`` beside everything it generates, so a frame
    it made three turns (or three threads) ago still knows it is "shot 2 start
    frame". Reading it here is what stops the next run describing pixels that
    already have a name. Best-effort and quiet: no ComfyUI, no record, no line.
    """
    if not isinstance(inputs, dict):
        return ""
    try:
        from src.utils.output_tags import role_of_canvas_file
        for key in ("image", "video", "file", "path", "filename", "audio"):
            val = inputs.get(key)
            if isinstance(val, str) and val.strip():
                role = role_of_canvas_file(val)
                if role:
                    return role
    except Exception:  # noqa: BLE001
        pass
    return ""


def _render_anchor(aid: str, atype: str, inputs: dict, tap: tuple | None = None,
                   role: str = "", title: str = "") -> str:
    """Human-readable description of one real-node anchor input.

    An agentY collector node is expanded to its listed on-disk file paths (available
    with no pre-run) so the agent can see/bind each file. An anchor whose *wire*
    carries a runtime tensor is rendered as the file(s)
    :mod:`src.utils.canvas_tap` rendered from it — otherwise the agent would see
    only a class name and have nothing to look at. Note this is a property of the
    wire, not the node: a ``LoadImage``'s MASK output gets tapped even though its
    IMAGE output is a file the agent can already read. Everything else lists its
    scalar params.
    """
    # The user said what this reference is FOR. It qualifies everything else on the
    # line, so it goes last, where it reads as the instruction it is.
    note = f'  ← USE THIS FOR: "{role.strip()}" (take only that from it)' if role.strip() else ""
    # Failing that: what the file itself says it is (agentY generated it and left a
    # record), then the node's title. Both are the user's or the agent's own words
    # about this input, and either beats making the next turn look at it again.
    if not note:
        was = _recorded_role(inputs)
        if not was:
            # A title the user (or the drop) gave the node. Agent-dropped nodes are
            # titled "agentY · <role>", so take the part after the mark — and drop
            # it entirely when it is just the filename again, which says nothing.
            stem = " ".join(str(title or "").split()).split(" · ", 1)[-1].strip()
            named = {str(v).strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
                     for v in (inputs or {}).values() if isinstance(v, str)}
            if stem and stem.lower() not in named:
                was = stem
        if was:
            note = f'  ← this is: "{was}"'
    if atype in _COLLECTOR_TYPES:
        files = inputs.get("files") if isinstance(inputs, dict) else None
        paths = [ln.strip().strip('"') for ln in str(files or "").splitlines() if ln.strip()]
        kind = "image" if atype == "AgentYImageCollector" else "video"
        if not paths:
            return f"node {aid} (agentY {kind} collector) — EMPTY (no files added yet){note}"
        return (f"node {aid} (agentY {kind} collector) — {len(paths)} {kind} file(s) already "
                f"on disk (use these paths directly, no run needed): " + "; ".join(paths) + note)
    params = ", ".join(f"{k}={v!r}" for k, v in (inputs or {}).items()) or "(no scalar inputs)"
    base = f"node {aid} ({atype}) inputs[{params}]"
    if tap:
        wire, paths = tap
        noun = "file" if len(paths) == 1 else "files"
        return (f"{base} — the {wire} output wired into this hook carries no file of its "
                f"own, so it was rendered to disk for you ({len(paths)} {noun}; these ARE "
                f"the content on that wire — use the path(s) directly): " + "; ".join(paths) + note)
    return base + note


def _trim(text: str, limit: int) -> str:
    """One line, at most *limit* characters, with an ellipsis when it was longer."""
    one = " ".join(str(text or "").split())
    return one if len(one) <= limit else one[:limit] + "…"


def _input_context(hook: dict, base_prompt: dict | None, hook_ids: set,
                   cached: dict | None = None) -> str:
    """Describe what feeds *hook* (its context inputs).

    An input wired from another HOOK is rendered as "the value you produce for
    hook N" — never a dump of that hook's own widgets — so a chained producer
    reuses the value it just wrote instead of mistaking the upstream hook's
    directive for content. Real-node inputs list their scalar params (or, for an
    agentY collector, its explicit file list).
    """
    def _from_hook(aid: str) -> str:
        done = str((cached or {}).get(aid) or "")
        if done:
            # The producer was memorized: hand over the value itself rather than a
            # promise to produce one, or the consumer waits for a turn that is
            # never going to happen.
            return f'the remembered value of hook {aid}: "{_trim(done, 300)}"'
        return f"the value you produce for hook {aid}"

    rows: list = []
    for aid, atype, inputs, tap, role, title, slot in _all_anchor_inputs(hook, base_prompt):
        text = _from_hook(aid) if aid in hook_ids else \
            _render_anchor(aid, atype, inputs, tap, role, title)
        rows.append((_slot_order(slot), _slot_label(slot), text))
    seen = {aid for aid, *_ in _all_anchor_inputs(hook, base_prompt)}
    for slot, hid in _chain_inputs(hook, hook_ids):
        if hid not in seen:
            rows.append((_slot_order(slot), _slot_label(slot), _from_hook(hid)))
    if not rows:
        return "no input wired"
    rows.sort(key=lambda r: r[0])
    # Name the slot each input arrived on when we know it — directives refer to
    # them ("the prompts in anchor_0"), and the mapping is not guessable from the
    # order alone once a hook gathers several.
    return "; ".join(f"{label}: {text}" if label else text for _o, label, text in rows)


def _split_targets(hook: dict, hook_ids: set) -> tuple[list, list]:
    """Output targets split into real node inputs and downstream HOOK anchors.

    A wire into another hook's anchor is a **chain handoff, not an input to fill**.
    Hook nodes are spliced out of the graph that actually runs, so a sweep aimed at
    one can never be applied — ``build_batch`` skips it and, when every target is a
    hook, the whole call comes back "no batch was produced". The consumer does not
    need anything written into it either: it already reads this hook's value from
    the block ("the value you produce for hook N"). So chain targets are described,
    never offered as something to fill.
    """
    real, chain = [], []
    for t in _output_targets(hook):
        (chain if t[0] in hook_ids else real).append(t)
    return real, chain


def _chain_note(chain: list) -> str:
    """Phrase the hooks that consume this hook's value, without naming an input."""
    if not chain:
        return ""
    ids = sorted({tid for tid, *_ in chain}, key=lambda s: (len(s), s))
    which = ", ".join(f"hook {i}" for i in ids)
    return f"{which} read the value you produce here as context"


def _target_context(hook: dict, hook_ids: set | None = None) -> str:
    """Describe where *hook*'s output goes — the producer's destination input(s).

    Only REAL node inputs; a wire into another hook is a chain handoff (see
    :func:`_split_targets`) and is reported separately.
    """
    targets, _chain = _split_targets(hook, hook_ids or set())
    if not targets:
        return ""
    # Anchors are the obvious things to connect a CONNECTION target to: the user
    # wired them into this hook as the material to choose from.
    anchors = [str(a.get("node_id")) for a in (hook.get("anchors") or [])
               if isinstance(a, dict) and a.get("node_id") is not None]
    parts: list = []
    for tid, ttype, tin, tintype, _ttitle in targets:
        tt = f", {tintype}" if tintype else ""
        slot = f"`{tin}`" if tin else "an input"
        line = f"node {tid} ({ttype})'s {slot} input{tt}"
        if is_connection_type(tintype):
            # This one carries a wire, not a value the agent can write. Say so at
            # the point of use, and name the nodes it can be connected to.
            pick = f" — connect one of {', '.join(anchors)}" if anchors else ""
            line += f" [CONNECTION: supply a node id{pick}, not a value]"
        parts.append(line)
    return "; ".join(parts)


def describe_hooks(hooks: list, base_prompt: dict | None = None) -> str:
    """Render the ``[CANVAS HOOKS]`` block injected into the orchestrator input.

    Hooks are **upstream producers**: each consumes its wired anchor inputs as
    context and produces value(s) for its ``out``, which the user wires into a real
    node input. Three purposes: an *inline_parameter* (producer) hook fills (or
    sweeps) the input its output is wired to; a *text* hook writes a single string
    the agent bakes there as an ``agentY text`` node; a *make_workflow* hook stands
    in for a workflow/script the agent generates. Hooks the user bypassed or muted on
    the canvas are filtered out client-side, so every hook below is active. Hooks are
    described in dependency order so a producer is handled before the hook that consumes it.
    """
    hooks = [h for h in (hooks or []) if isinstance(h, dict)]
    if not hooks:
        return ""
    hooks = _order_by_dependency(hooks)
    hook_id_set = _hook_ids(hooks)
    # A memorizing hook whose inputs haven't changed was answered before, and the
    # pipeline has already put that answer back into the graph. It is reported, not
    # assigned: from here on "hooks" means the ones still to be done, so nothing
    # downstream — the run plan, the sitrep, the work lists — offers it as work.
    all_hooks = hooks
    cached_map = {str(h.get("hook_node_id")): str((h.get("_cached") or {}).get("value") or "")
                  for h in hooks if h.get("_cached")}
    cached_hooks = [h for h in hooks if h.get("_cached")]
    hooks = [h for h in hooks if not h.get("_cached")]
    hook_id_set = _hook_ids(all_hooks)
    text_hooks = [h for h in hooks if _is_text(h)]
    standin_hooks = [h for h in hooks if _is_standin(h)]
    iterate_hooks = [h for h in hooks if _is_iterate(h)]
    general_hooks = [h for h in hooks if _is_general(h)]
    qa_hooks = [h for h in hooks if _is_qa(h)]
    directive_hooks = [h for h in hooks
                       if not _is_standin(h) and not _is_text(h)
                       and not _is_iterate(h) and not _is_general(h)
                       and not _is_qa(h)]

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

    if cached_hooks:
        lines.append(
            "\nALREADY DONE (remembered — these hooks have 'memorize' on and nothing "
            "feeding them has changed since they last ran, so their value is already "
            "back in the graph). Do NOT redo them, do not re-read their inputs, and do "
            "not describe their anchors again:"
        )
        for h in cached_hooks:
            hid = h.get("hook_node_id")
            c = h.get("_cached") or {}
            where = (", filled node(s) " + ", ".join(c.get("targets") or [])
                     if c.get("targets") else ", not wired into a node input")
            when = f" [{c['when']}]" if c.get("when") else ""
            lines.append(f'- hook {hid}{when}{where} → "{_trim(c.get("value"), 400)}"')
        if not hooks:
            lines.append(
                "  That is every hook on this graph. There is no value left to produce: "
                "if the user asked for a run, call apply_canvas_hooks(resolutions=[]) once "
                "to run the canvas exactly as it now stands; otherwise just answer them. "
                "To make a hook produce a fresh value, the user turns its 'memorize' "
                "toggle off (or changes what feeds it)."
            )

    # When hooks feed each other, spell out the order so producers are done first.
    if any(_hook_predecessors(h, hook_id_set) for h in hooks):
        order = " → ".join(str(h.get("hook_node_id")) for h in hooks)
        lines.append(
            f"\nPROCESS ORDER (producers first): {order}. A hook whose input is 'the "
            "value you produce for hook N' consumes another hook's output — produce hook "
            "N FIRST and reuse exactly what you wrote as this hook's context; do NOT "
            "re-read it from the graph."
        )

    # When a directive gates on an earlier step's outcome, say up front which hooks
    # have to be run (not queued) for that check to be possible at all — then what
    # the turn is about to assume, while it can still be corrected.
    lines.extend(plan_lines(hooks))
    # What cannot work, before anything runs — the graph and the node schemas
    # together, which is where a contradiction between wiring and directive shows.
    try:
        from src.utils.preflight import lines as _preflight
        lines.extend(_preflight(hooks, base_prompt))
    except Exception:  # noqa: BLE001 — a check must never cost the turn
        pass
    lines.extend(sitrep_lines(hooks, base_prompt))

    if directive_hooks:
        lines.append(
            "\nPRODUCER hooks — each produces value(s) for the node input its OUTPUT is "
            "wired to (shown as 'feeds …' below); that wired input is the target — do NOT "
            "guess a node from the prose, and do NOT call prepare_workflow, run_research or "
            "assemble a template. The node that does the generating is ALREADY on the "
            "canvas at the other end of the wire; a directive asking for many outputs (\"one "
            "pass per prompt\", \"three consecutive runs\") is a SWEEP of that node, not a "
            "workflow to build — give apply_canvas_hooks one value per run. "
            "Two ways to produce, by how many values the directive asks for:\n"
            "  • ONE value (e.g. a single composed prompt) → write it and call "
            'place_canvas_text(hook_node_id="<hook id>", text="<value>") — it delivers the '
            "value to the target input (injected at run time if the hook is kept live, or "
            "baked in if its 'bake' switch is on — the hook's own setting) and drops an "
            "'agentY text' node.\n"
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
            ctx = _input_context(h, base_prompt, hook_id_set, cached_map)
            real, chain = _split_targets(h, hook_id_set)
            tgt = _target_context(h, hook_id_set)
            if tgt:
                also = f" ({_chain_note(chain)})" if chain else ""
                lines.append(
                    f'- PRODUCER hook {hid} (context: {ctx}) feeds {tgt}{also} — produce the '
                    f'value(s) for that input → "{directive}"'
                )
            elif chain:
                # Every consumer is another hook, so there is no graph input to fill
                # or sweep: apply_canvas_hooks would have nothing to apply. Write the
                # value and hand it on.
                lines.append(
                    f'- PRODUCER hook {hid} (context: {ctx}) — CHAIN ONLY: {_chain_note(chain)}, '
                    f'and its output reaches no real node input. There is nothing to fill or '
                    f'sweep here: WRITE the value and deliver it with '
                    f'place_canvas_text(hook_node_id="{hid}", text="<value>"). Do NOT call '
                    f'apply_canvas_hooks for this hook and do NOT build a workflow for it '
                    f'→ "{directive}"'
                )
            else:
                lines.append(
                    f'- PRODUCER hook {hid} (context: {ctx}) — OUTPUT UNWIRED: no target '
                    f'input. Ask the user to wire this hook\'s output into the node input it '
                    f'should fill. Directive: "{directive}"'
                )

    if general_hooks:
        lines.append(
            "\nGENERAL-REQUEST hook(s) — a FREE-FORM instruction. Treat the directive as "
            "an ordinary request from the user, with any wired anchor(s) as the provided "
            "input(s)/context and THIS graph already captured. Decide the right action "
            "yourself — answer a question, generate or edit media, run a workflow, compute "
            "a value — whatever best fulfils it; you are NOT restricted to "
            "apply_canvas_hooks/place_canvas_text here. If it yields MEDIA via a ComfyUI "
            "workflow, generate/run via the normal generation contract and let the outputs "
            "stage onto the canvas as loader nodes (use a wired image/video as input — "
            "upload_image it and bind it). If instead you fulfil it with an ASYNC MCP "
            "generator (e.g. Magnific), there is nothing to stage — just queue it and share "
            "the returned URL; the finished asset is downloaded and dropped onto the canvas "
            "automatically when ready. If it produces ONE value for the node input this hook's output feeds "
            '(shown as "feeds …"), deliver it with place_canvas_text(hook_node_id="<id>", '
            'text="<value>"). If it\'s just a question, answer it in chat.'
        )
        for h in general_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            ctx = _input_context(h, base_prompt, hook_id_set, cached_map)
            tgt = _target_context(h, hook_id_set)
            _real, chain = _split_targets(h, hook_id_set)
            where = (f" feeds {tgt}" if tgt else
                     f" ({_chain_note(chain)})" if chain else " (output unwired)")
            lines.append(f'- GENERAL hook {hid} (context: {ctx}){where} — "{directive}"')

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
            tgt = _target_context(h, hook_id_set)
            ctx = _input_context(h, base_prompt, hook_id_set, cached_map)
            prompt_where = (f"prompt → {tgt}" if tgt else
                            "prompt target UNWIRED — ask the user to wire this hook's OUTPUT "
                            "into the prompt node's text input")
            fb_where = (f"feedback image ← {ctx}" if ctx and ctx != "no input wired" else
                        "feedback node UNWIRED — ask the user to wire the LoadImage node's "
                        "image output into this hook's anchor")
            tail = f' — "{directive}"' if directive else ""
            lines.append(f"- ITERATE hook {hid}: {prompt_where}; {fb_where}{tail}")

    if qa_hooks:
        lines.append(
            "\nQA hook(s) — these are NOT work for you. Each carries the user's QUALITY "
            "BRIEFING for this graph: its directive is the checklist and its wired "
            "anchors are reference/mood images. A separate QA agent applies it to every "
            "image/video the run produces, AFTER generation — you do not have to check "
            "anything yourself, and you must NOT treat the anchors as inputs to a "
            "workflow, place_canvas_text them, or apply_canvas_hooks them. Note the "
            "criteria while you write prompts (satisfying them up front beats being sent "
            "back), then carry on with the rest of the request as usual. If the QA agent "
            "later reports a failure you will be given the failed criteria and asked to "
            "adjust and re-run:"
        )
        for h in qa_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            refs = sum(len((a.get("tapped") or [])) or 1
                       for a in (h.get("anchors") or []) if isinstance(a, dict))
            ref_txt = f"{refs} reference input(s) wired" if refs else "no reference images wired"
            lines.append(f'- QA hook {hid} ({ref_txt}) — criteria: "{directive}"')

    if text_hooks:
        lines.append(
            "\nTEXT hooks — each produces a single WRITTEN string (not media). WRITE the "
            "answer yourself (activate a relevant writing skill if it helps). Do NOT "
            "generate images/video, do NOT call apply_canvas_hooks, and do NOT build or "
            "run a workflow. Use the wired context as the SUBJECT of the answer. When the "
            'answer is ready, call place_canvas_text(hook_node_id="<id>", text="<answer>") '
            "ONCE per hook — it delivers the string to the input the hook's output feeds "
            "(shown as 'feeds …'; injected at run time if the hook is kept live, or baked in "
            "if its 'bake' switch is on — the hook's own setting) and drops an 'agentY text' "
            "node. The answer "
            "also streams into the chat:"
        )
        for h in text_hooks:
            hid = h.get("hook_node_id")
            directive = str(h.get("directive", "") or "").strip()
            ctx = _input_context(h, base_prompt, hook_id_set, cached_map)
            tgt = _target_context(h, hook_id_set)
            _real, chain = _split_targets(h, hook_id_set)
            where = (f" feeds {tgt}" if tgt else
                     f" ({_chain_note(chain)})" if chain else
                     " (output unwired — answer streams to chat only)")
            lines.append(f'- TEXT hook {hid} (context: {ctx}){where} — write & place → "{directive}"')

    if standin_hooks:
        chains = _order_standin_chains(standin_hooks)
        singles = [c[0] for c in chains if len(c) == 1]
        multis = [c for c in chains if len(c) > 1]

        def _input_desc(h: dict) -> str:
            anchors = _all_anchor_inputs(h, base_prompt)
            if not anchors:
                return "no input wired — treat the prompt as text-to-media"
            parts = [_render_anchor(aid, atype, inputs, tap, role, title)
                     for aid, atype, inputs, tap, role, title, _slot in anchors]
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
                "\nBAKE TO CANVAS — one or more make_workflow hooks above has its 'bake' switch "
                "ON. After you have GENERATED and validated each such stage's workflow, do "
                "NOT stop at running it: call bake_hooks_to_canvas(stages=[…]) to nest each "
                "generated workflow into a ComfyUI subgraph whose inputs/outputs MATCH that "
                "hook's slots, place the subgraphs on the same canvas, and wire them to "
                "mirror the hook chain — baking the multi-step task into a reusable native "
                "workflow the user can re-run without you. For each baked stage pass: "
                "workflow_path (the generated workflow), a SHORT `name` — 2-5 words for "
                "what the stage DOES (\"Upscale 2x + grain\", \"Animate the scene\"), never "
                "the directive, which is a paragraph and would be the label on a collapsed "
                "node — hook_node_id, exposed inputs "
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
