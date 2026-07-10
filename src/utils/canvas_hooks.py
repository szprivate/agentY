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
from pathlib import Path

_HOOK_CLASS = "AgentYHook"

IMG_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff"}
VID_EXTS = {"mp4", "mov", "webm", "mkv", "avi"}


def _anchor_links(inputs: dict) -> list:
    """Return each wired ``anchor`` link ``[node_id, slot]`` on a hook, in slot
    order. The anchor input auto-grows, so its names are ``anchor``/``anchor0``/… ."""
    def _idx(name: str) -> int:
        suf = name[len("anchor"):]
        return int(suf) if suf.isdigit() else -1  # bare "anchor" sorts first

    keys = [k for k in (inputs or {})
            if k == "anchor" or (k.startswith("anchor") and k[len("anchor"):].isdigit())]
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


def build_batch(base_prompt: dict, resolutions: list, cap: int = 25) -> tuple[list[dict], list[str]]:
    """Expand *base_prompt* into a mutated batch from *resolutions*.

    Each resolution mutates one node's input across a list of values; the batch is
    the Cartesian product across all resolutions, capped at *cap*. Returns
    ``(prompts, notes)`` where *notes* explains any skips/truncation.
    """
    notes: list[str] = []
    axes: list[tuple[str, str, list]] = []
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
        values = _resolve_values(res)
        if not values:
            notes.append(f"node {nid}.{param}: no values resolved — skipped")
            continue
        axes.append((nid, param, values))

    if not axes:
        return [], (notes or ["no valid resolutions were supplied"])

    combos: list[list] = [[]]
    for (_nid, _param, values) in axes:
        combos = [c + [v] for c in combos for v in values]
    total = len(combos)
    if total > cap:
        notes.append(f"batch of {total} exceeded the cap of {cap}; truncated to {cap}")
        combos = combos[:cap]

    prompts: list[dict] = []
    for combo in combos:
        p = copy.deepcopy(base_prompt)
        for (nid, param, _values), val in zip(axes, combo):
            node = p.get(nid)
            if isinstance(node, dict):
                node.setdefault("inputs", {})[param] = val
        prompts.append(p)
    return prompts, notes


_STANDIN_PURPOSES = {"workflow-standin", "workflow_standin", "standin", "workflow"}


def _is_standin(hook: dict) -> bool:
    """True if *hook* is a workflow-standin (vs. an annotation directive)."""
    return str(hook.get("purpose", "directive") or "directive").strip().lower() in _STANDIN_PURPOSES


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


def describe_hooks(hooks: list, base_prompt: dict | None = None) -> str:
    """Render the ``[CANVAS HOOKS]`` block injected into the orchestrator input.

    Hooks come in two purposes (set on the node): *directive* hooks annotate an
    anchor node and are run by expanding the captured graph via
    ``apply_canvas_hooks``; *workflow-standin* hooks are self-contained generation
    requests the agent fulfils by generating and running a workflow/script. Hooks
    the user toggled to *ignore* are filtered out client-side before they reach
    here, so every hook below is active.
    """
    hooks = [h for h in (hooks or []) if isinstance(h, dict)]
    if not hooks:
        return ""
    directive_hooks = [h for h in hooks if not _is_standin(h)]
    standin_hooks = [h for h in hooks if _is_standin(h)]

    lines = [
        "[CANVAS HOOKS — the user's ON-CANVAS graph carries hook annotations (below) "
        "and is already captured. IF the user is asking you to run/execute the "
        "workflow, act on the hooks as described below. If the user's message is "
        "unrelated (a question or a different request), answer that and ignore these "
        "hooks.]"
    ]

    if directive_hooks:
        lines.append(
            "\nDIRECTIVE hooks — expand and run THIS captured graph (do NOT assemble a "
            "template or call run_research). Interpret each directive against its "
            "anchor node and call apply_canvas_hooks(resolutions=[…]) ONCE to run the "
            "batch. Each resolution targets an anchor node id below and one of its "
            "inputs:"
        )
        for h in directive_hooks:
            anchors = _all_anchor_inputs(h, base_prompt)
            directive = str(h.get("directive", "") or "").strip()
            mode = h.get("mode", "auto")
            if not anchors:
                lines.append(
                    f'- UNWIRED hook: "{directive}" (mode={mode}). No anchor node — ask '
                    "the user to wire it to a node's output, or apply globally only if "
                    "unambiguous."
                )
            else:
                # One line per anchor — the directive applies to each wired node.
                for aid, atype, inputs in anchors:
                    params = ", ".join(f"{k}={v!r}" for k, v in inputs.items()) or "(no scalar inputs)"
                    lines.append(f'- Node {aid} ({atype}) inputs[{params}] ← "{directive}" (mode={mode})')

    if standin_hooks:
        chains = _order_standin_chains(standin_hooks)
        singles = [c[0] for c in chains if len(c) == 1]
        multis = [c for c in chains if len(c) > 1]

        def _input_desc(h: dict) -> str:
            anchors = _all_anchor_inputs(h, base_prompt)
            if not anchors:
                return "no input wired — treat the prompt as text-to-media"
            parts = []
            for aid, atype, inputs in anchors:
                params = ", ".join(f"{k}={v!r}" for k, v in inputs.items()) or "(no scalar inputs)"
                parts.append(f"node {aid} ({atype}) inputs[{params}]")
            if len(parts) == 1:
                return f"input from {parts[0]}"
            return "inputs from " + "; ".join(parts)

        if singles:
            lines.append(
                "\nWORKFLOW-STANDIN hooks — each is a self-contained generation request. "
                "For each one, GENERATE a ComfyUI workflow that fulfils the prompt (or, "
                "when a workflow doesn't fit, a Python script written into the scripts "
                "dir from get_agent_output_dirs()), then run it via the normal "
                "generation contract — signal_workflow_ready for a workflow, or run the "
                "script — and let the outputs stage onto the canvas as loader nodes. Do "
                "NOT call apply_canvas_hooks for these. If an anchor is wired, its output "
                "is the INPUT to what you generate (e.g. upload that image/video and bind "
                "it to the loader); if nothing is wired, treat the prompt as a "
                "text-to-media request. Media routing (agent/images, agent/videos, …) is "
                "enforced automatically:"
            )
            for h in singles:
                prompt = str(h.get("directive", "") or "").strip()
                lines.append(f'- STANDIN, {_input_desc(h)} — generate & run → "{prompt}"')

        if multis:
            lines.append(
                "\nWORKFLOW-STANDIN CHAINS — a chain of hooks wired output→input. Run the "
                "stages STRICTLY IN ORDER, feeding each stage's OUTPUT as the next "
                "stage's INPUT. For each stage: GENERATE a workflow (or script) from its "
                "prompt, then run it with run_workflow_now(workflow_path) — NOT "
                "signal_workflow_ready, because you need each stage's output file to "
                "build the next. Take the output path it returns, upload_image it, and "
                "bind it to the next stage's input loader. Stage 1's input is its wired "
                "anchor (if any), else text-to-media; the final stage's output is the "
                "result. Do NOT call apply_canvas_hooks for these:"
            )
            for ci, chain in enumerate(multis, 1):
                head_in = _input_desc(chain[0])
                lines.append(f"  Chain {ci} (stage 1 {head_in}):")
                for si, h in enumerate(chain, 1):
                    prompt = str(h.get("directive", "") or "").strip()
                    lines.append(f'    {si}. "{prompt}"')

    lines.append("")
    return "\n".join(lines)
