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


def splice_hook_nodes(prompt: dict) -> tuple[dict, list[str]]:
    """Return ``(clean_prompt, removed_ids)``.

    Removes every ``AgentYHook`` node from an API-format *prompt*. For an inline
    hook (its output feeds a downstream node) each downstream input is rewired to
    the hook's own ``anchor`` source so the graph stays connected. A dangling hook
    (nothing consumes it) is simply dropped. Works whether or not ComfyUI's
    ``graphToPrompt`` already pruned the hook.
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
        anchor = (node.get("inputs") or {}).get("anchor")
        src = anchor if (isinstance(anchor, list) and len(anchor) == 2) else None
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


def describe_hooks(hooks: list, base_prompt: dict | None = None) -> str:
    """Render the ``[CANVAS HOOKS]`` block injected into the orchestrator input."""
    hooks = [h for h in (hooks or []) if isinstance(h, dict)]
    if not hooks:
        return ""
    lines = [
        "[CANVAS HOOKS — the user's ON-CANVAS graph has hook annotations (below) and "
        "is already captured. IF the user is asking you to run/execute the workflow, "
        "run THIS graph — do NOT assemble a template or call run_research — by "
        "interpreting each directive against its anchor node and calling "
        "apply_canvas_hooks(resolutions=[…]) ONCE to expand and run the batch. If the "
        "user's message is unrelated (a question or a different request), answer that "
        "and ignore these hooks. Each resolution targets an anchor node id below and "
        "one of its inputs:]"
    ]
    for h in hooks:
        aid = h.get("anchor_node_id")
        atype = h.get("anchor_type") or "?"
        directive = str(h.get("directive", "") or "").strip()
        mode = h.get("mode", "auto")
        inputs: dict = {}
        if base_prompt and aid is not None and str(aid) in base_prompt:
            raw = (base_prompt[str(aid)].get("inputs") or {})
            inputs = {k: v for k, v in raw.items() if not isinstance(v, list)}
        elif isinstance(h.get("anchor_widgets"), dict):
            inputs = h["anchor_widgets"]
        params = ", ".join(f"{k}={v!r}" for k, v in inputs.items()) or "(no scalar inputs)"
        if aid is None:
            lines.append(
                f'- UNWIRED hook: "{directive}" (mode={mode}). No anchor node — ask '
                "the user to wire it to a node's output, or apply globally only if "
                "unambiguous."
            )
        else:
            lines.append(f'- Node {aid} ({atype}) inputs[{params}] ← "{directive}" (mode={mode})')
    lines.append("")
    return "\n".join(lines)
