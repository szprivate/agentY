"""Render a hook anchor's *runtime tensor* to a file the agent can actually look at.

A canvas hook reads its wired anchor as context. When that anchor is a loader
(``LoadImage``, a video loader, an agentY collector) the file is already named in
the node's own widgets, so the agent sees it with no run at all — that is why the
hook block could show images only for a loader wired straight into the hook.
Anything else — a ``VAEDecode``, an upscaler, an ``ImageBlend``, a mask op — holds
its value only as a **tensor that exists during a run**, so the hook block could
name the node but never show the picture.

This module closes that gap. For every such anchor it

1. trims the captured graph to the anchor's **ancestors** (exactly what is needed
   to produce that one wire, nothing downstream of it),
2. appends a preview/save node to the exact output slot the hook is wired to,
   converting ``MASK``/``LATENT`` to an image on the way, and
3. runs the lot as **one** extra prompt, so every tap of a turn shares a single
   execution (and any shared upstream work is computed once).

The tap prompt is a throwaway copy: nothing on the user's canvas is touched, and
because the trim drops everything downstream of the wire, the user's own savers
never fire. Images land in ComfyUI's ``temp`` space; a tapped ``VIDEO`` has to go
through ``SaveVideo`` (ComfyUI has no preview-video node), so those land under an
``agentY_tap/`` subfolder of the output dir. Either way the file is fetched back
over ``/view`` into a per-turn temp dir, so the path stays valid for the whole
turn even though ComfyUI may recycle its scratch.

Cost: when the graph has already been run, ComfyUI's execution cache makes the tap
near-instant. When it has not, this really does render the upstream branch — that
is the price of seeing what is on the wire, and why it is capped
(``AGENTY_MAX_HOOK_TAPS``) and can be switched off entirely (``AGENTY_HOOK_TAP=0``
/ ``hook_tap_tensors`` in settings).
"""
from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Wire types worth materialising. Everything else (MODEL, CLIP, CONDITIONING, a
# plain STRING/INT) is either meaningless as a picture or already readable from
# the node's widgets in the [CANVAS HOOKS] block.
TAPPABLE_TYPES = {"IMAGE", "MASK", "LATENT", "VIDEO"}

# Only these are ever "already visible from widgets" (see ``_is_already_visible``):
# a LoadImage's MASK output is NOT the file its `image` widget names, so a mask or
# latent tap is never skipped just because the node happens to hold a filename.
_WIDGET_VISIBLE_TYPES = {"IMAGE", "VIDEO"}

_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff",
               ".mp4", ".mov", ".webm", ".mkv", ".avi"}

# agentY collector nodes already render as an explicit on-disk file list, so their
# wires never need a run to be understood.
_COLLECTOR_TYPES = {"AgentYImageCollector", "AgentYVideoCollector"}

# Placeholder for "the wire this chain taps" — replaced with a real link when the
# chain is materialised into the prompt (see ``_append_chain``).
_SRC = "\x00tap-src"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def taps_enabled() -> bool:
    """Whether tensor anchors are materialised at all.

    ``AGENTY_HOOK_TAP`` wins when set; otherwise ``hook_tap_tensors`` in settings
    (default on). Off means the hook block still describes the anchor node — the
    agent just cannot see what is on the wire, i.e. the behaviour before taps.
    """
    env = os.environ.get("AGENTY_HOOK_TAP")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("hook_tap_tensors", True))
    except Exception:  # noqa: BLE001 — never let settings break a turn
        return True


@dataclass(frozen=True)
class Tap:
    """One wire to materialise: *slot* of node *node_id*.

    ``anchors`` are every hook anchor reading that same wire — they all get the
    produced paths, but the wire is only rendered once.
    """
    node_id: str
    slot: int
    out_type: str
    anchors: list


# ── discovering what is on a wire ────────────────────────────────────────────────

_OBJECT_INFO_CACHE: dict[str, list[str]] = {}


def _declared_output_types(class_type: str) -> list[str]:
    """Output types ComfyUI declares for *class_type*, cached per process.

    The fallback for frontends that don't send the anchor's source type (the older
    sidebar, or a wildcard/reroute slot LiteGraph never resolved). Returns [] when
    ComfyUI is unreachable or the class is unknown — the caller then skips the tap
    rather than guessing.
    """
    if class_type in _OBJECT_INFO_CACHE:
        return _OBJECT_INFO_CACHE[class_type]
    types: list[str] = []
    try:
        from agenty_core.utils.comfyui_client import get_client
        info = get_client().get(f"/object_info/{class_type}")
        entry = (info or {}).get(class_type) if isinstance(info, dict) else None
        raw = (entry or {}).get("output") if isinstance(entry, dict) else None
        if isinstance(raw, list):
            types = [str(t).upper() for t in raw]
    except Exception as exc:  # noqa: BLE001
        logger.debug("canvas_tap: could not read /object_info/%s: %s", class_type, exc)
    _OBJECT_INFO_CACHE[class_type] = types
    return types


def _wire_type(anchor: dict, prompt: dict, node_id: str, slot: int) -> str:
    """The type carried by the tapped wire, upper-cased ('' when undecidable)."""
    t = str(anchor.get("from_output_type") or "").strip().upper()
    if t and t not in ("*", "ANY", "WILDCARD"):
        return t
    class_type = str((prompt.get(node_id) or {}).get("class_type") or "")
    if not class_type:
        return ""
    declared = _declared_output_types(class_type)
    return declared[slot] if 0 <= slot < len(declared) else ""


def _is_already_visible(anchor: dict, resolver) -> bool:
    """True when the agent can already see this anchor's content without a run.

    That means an agentY collector (rendered as its file list) or any node holding
    a widget value that resolves to a media file on disk — which is exactly what
    makes a ``LoadImage``/video loader readable today. *resolver* is the caller's
    ``value, kind -> abs path | None`` (the server's ``_resolve_media_ref``); with
    none supplied only the collector shortcut applies.
    """
    if str(anchor.get("type") or "") in _COLLECTOR_TYPES:
        return True
    if resolver is None:
        return False
    for value in (anchor.get("widgets") or {}).values():
        if not isinstance(value, str) or not value.strip():
            continue
        if Path(value.strip().strip('"')).suffix.lower() not in _MEDIA_EXTS:
            continue  # a checkpoint/LoRA name is not something to show
        try:
            if resolver(value, ""):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def plan_taps(hooks: list, prompt: dict, *, resolver=None, cap: int | None = None) -> list[Tap]:
    """Which anchor wires in *hooks* need materialising against *prompt*.

    *prompt* must be the API-format graph with hook nodes already spliced out (see
    :func:`src.utils.canvas_hooks.splice_hook_nodes`) so a hook→hook wire never
    looks like a tensor. Anchors pointing at a node that isn't in the prompt
    (muted, bypassed, or another hook) are skipped, as are wires the agent can
    already read from widgets. Deduped per (node, slot) so two hooks sharing one
    upstream node cost one tap.
    """
    if cap is None:
        cap = _env_int("AGENTY_MAX_HOOK_TAPS", 4)
    out: list[Tap] = []
    seen: dict[tuple[str, int], Tap] = {}
    for hook in (hooks or []):
        if not isinstance(hook, dict):
            continue
        for anchor in (hook.get("anchors") or []):
            if not isinstance(anchor, dict) or anchor.get("node_id") is None:
                continue
            node_id = str(anchor["node_id"])
            if node_id not in prompt:
                continue
            try:
                slot = int(anchor.get("from_output_slot") or 0)
            except (TypeError, ValueError):
                slot = 0
            wire = _wire_type(anchor, prompt, node_id, slot)
            if wire not in TAPPABLE_TYPES:
                continue
            if wire in _WIDGET_VISIBLE_TYPES and _is_already_visible(anchor, resolver):
                continue
            key = (node_id, slot)
            if key in seen:
                seen[key].anchors.append(anchor)  # same wire, another hook
                continue
            if len(out) >= cap:
                logger.info("canvas_tap: more than %d tensor anchor(s) — tapping the first %d",
                            cap, cap)
                return out
            tap = Tap(node_id=node_id, slot=slot, out_type=wire, anchors=[anchor])
            seen[key] = tap
            out.append(tap)
    return out


# ── building the trimmed tap prompt ─────────────────────────────────────────────

def _ancestors(prompt: dict, root: str) -> set[str]:
    """*root* plus every node it transitively depends on (its input closure)."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in prompt:
            continue
        seen.add(nid)
        for value in ((prompt[nid] or {}).get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2:
                stack.append(str(value[0]))
    return seen


def _find_vae_link(prompt: dict) -> list | None:
    """A ``[node_id, slot]`` producing a VAE, for decoding a tapped LATENT.

    Reuses whatever the graph itself decodes with — the ``vae`` input of an existing
    VAEDecode/VAEEncode — so the preview matches the pipeline the user built. Falls
    back to any VAE loader present. None when the graph has no VAE at all (that
    latent then isn't tapped rather than being decoded with a guess).
    """
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "").startswith("VAEDecode") or \
                str(node.get("class_type") or "").startswith("VAEEncode"):
            vae = (node.get("inputs") or {}).get("vae")
            if isinstance(vae, list) and len(vae) == 2:
                return list(vae)
    for nid, node in prompt.items():
        if isinstance(node, dict) and "VAELoader" in str(node.get("class_type") or ""):
            return [nid, 0]
    return None


def _chain_spec(prompt: dict, tap: Tap, frames: int) -> tuple[list, list[str]] | None:
    """``([(class_type, inputs), …], extra_roots)`` turning *tap*'s wire into a file.

    Each entry's ``_SRC`` input is bound to the previous entry's output (the first
    to the tapped wire itself); the last entry is the output node whose results are
    collected. *extra_roots* are nodes the chain itself depends on and which must
    therefore survive the trim (the VAE for a latent). None when the type can't be
    rendered with what this graph provides.
    """
    batch = ("ImageFromBatch", {"image": _SRC, "batch_index": 0, "length": frames})
    preview = ("PreviewImage", {"images": _SRC})
    if tap.out_type == "IMAGE":
        return [batch, preview], []
    if tap.out_type == "MASK":
        return [("MaskToImage", {"mask": _SRC}), batch, preview], []
    if tap.out_type == "LATENT":
        vae = _find_vae_link(prompt)
        if vae is None:
            logger.info("canvas_tap: node %s outputs LATENT but the graph has no VAE — skipped",
                        tap.node_id)
            return None
        return [("VAEDecode", {"samples": _SRC, "vae": list(vae)}), batch, preview], [str(vae[0])]
    if tap.out_type == "VIDEO":
        return [("SaveVideo", {"video": _SRC, "filename_prefix": "agentY_tap/tap",
                               "format": "auto", "codec": "auto"})], []
    return None


def _free_ids(prompt: dict, count: int) -> list[str]:
    """*count* node ids not used by *prompt*, numeric so nodes that coerce their
    ``unique_id`` hidden input keep working."""
    highest = 0
    for key in prompt:
        try:
            highest = max(highest, int(key))
        except (TypeError, ValueError):
            continue
    out: list[str] = []
    candidate = highest + 1
    while len(out) < count:
        if str(candidate) not in prompt:
            out.append(str(candidate))
        candidate += 1
    return out


def build_tap_prompt(prompt: dict, taps: list[Tap], *, frames: int | None = None
                     ) -> tuple[dict, dict[str, Tap]]:
    """``(tap_prompt, sink_id -> Tap)`` — one throwaway graph rendering every tap.

    The graph is *prompt* trimmed to the union of the taps' ancestors, plus each
    tap's conversion chain. Trimming is what keeps this cheap and safe: samplers,
    savers and other branches downstream of (or unrelated to) the tapped wires are
    never part of the run, so tapping an input image can't kick off the user's
    whole pipeline or write into their output folder.
    """
    if frames is None:
        frames = max(1, _env_int("AGENTY_HOOK_TAP_FRAMES", 4))

    chains: list[tuple[Tap, list]] = []
    keep: set[str] = set()
    for tap in taps:
        spec = _chain_spec(prompt, tap, frames)
        if spec is None:
            continue
        nodes, extra_roots = spec
        chains.append((tap, nodes))
        keep |= _ancestors(prompt, tap.node_id)
        for root in extra_roots:
            keep |= _ancestors(prompt, root)
    if not chains:
        return {}, {}

    tap_prompt = {nid: copy.deepcopy(node) for nid, node in prompt.items() if nid in keep}
    sinks: dict[str, Tap] = {}
    for tap, nodes in chains:
        ids = _free_ids(tap_prompt, len(nodes))
        prev: list = [tap.node_id, tap.slot]
        for new_id, (class_type, inputs) in zip(ids, nodes):
            resolved = {k: (list(prev) if v == _SRC else v) for k, v in inputs.items()}
            tap_prompt[new_id] = {"class_type": class_type, "inputs": resolved}
            prev = [new_id, 0]
        sinks[ids[-1]] = tap
    return tap_prompt, sinks


# ── running it ──────────────────────────────────────────────────────────────────

def _submit(tap_prompt: dict) -> str:
    from agenty_core.utils.comfyui_client import get_client

    client = get_client()
    payload: dict = {"prompt": tap_prompt}
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}
    result = client.post("/prompt", json_data=payload)
    if isinstance(result, dict) and result.get("prompt_id"):
        return str(result["prompt_id"])
    raise RuntimeError(f"unexpected /prompt response: {result!r}")


def _await_history(prompt_id: str, timeout: float) -> dict:
    """Block until ComfyUI finishes *prompt_id*, returning its history entry.

    Polls rather than opening a WebSocket: the tap runs on the request thread
    before the turn starts, and a poll needs no client_id handshake to be tied to
    the right prompt.
    """
    from agenty_core.utils.comfyui_client import get_client

    client = get_client()
    deadline = time.monotonic() + timeout
    delay = 0.25
    while time.monotonic() < deadline:
        try:
            hist = client.get(f"/history/{prompt_id}")
        except Exception as exc:  # noqa: BLE001 — a blip mustn't abort the wait
            logger.debug("canvas_tap: history poll failed: %s", exc)
            hist = None
        entry = (hist or {}).get(prompt_id) if isinstance(hist, dict) else None
        if isinstance(entry, dict) and entry.get("outputs") is not None:
            status = entry.get("status") or {}
            if not status or status.get("completed") or status.get("status_str") == "success":
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(_history_error(status))
        time.sleep(delay)
        delay = min(delay * 1.4, 1.5)
    raise TimeoutError(f"ComfyUI did not finish the tap run within {timeout:.0f}s")


def _history_error(status: dict) -> str:
    """The most useful line out of a failed run's history messages."""
    for name, payload in reversed(status.get("messages") or []):
        if name == "execution_error" and isinstance(payload, dict):
            return (f"{payload.get('node_type', '?')}: "
                    f"{payload.get('exception_message', 'execution error')}")
    return "the tap run failed in ComfyUI"


def _fetch(record: dict, dest_dir: Path) -> str | None:
    """Pull one produced file to a stable local path under *dest_dir*.

    ``PreviewImage`` writes into ComfyUI's temp space, which is not addressable
    from here (and is recycled), so everything is fetched over ``/view`` — which
    also means a ComfyUI on another host works unchanged.
    """
    from agenty_core.utils.comfyui_client import get_client

    filename = str(record.get("filename") or "")
    if not filename:
        return None
    params = {"filename": filename, "type": str(record.get("type") or "temp")}
    subfolder = str(record.get("subfolder") or "")
    if subfolder:
        params["subfolder"] = subfolder
    try:
        resp = get_client().get("/view", params=params, raw=True)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        if dest.exists():  # two taps can produce identically-named temp files
            dest = dest_dir / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
        dest.write_bytes(resp.content)  # type: ignore[attr-defined]
        return str(dest)
    except Exception as exc:  # noqa: BLE001
        logger.warning("canvas_tap: could not fetch %s: %s", filename, exc)
        return None


def _collect(entry: dict, sinks: dict[str, Tap], dest_dir: Path) -> dict[str, list[str]]:
    """``sink_id -> [local paths]`` from a finished run's history entry."""
    out: dict[str, list[str]] = {}
    for node_id, node_out in (entry.get("outputs") or {}).items():
        if str(node_id) not in sinks or not isinstance(node_out, dict):
            continue
        paths: list[str] = []
        for key in ("images", "gifs", "videos", "audio"):
            for record in node_out.get(key) or []:
                if isinstance(record, dict):
                    path = _fetch(record, dest_dir)
                    if path:
                        paths.append(path)
        if paths:
            out[str(node_id)] = paths
    return out


def materialize_hook_tensors(hooks: list, prompt: dict, *, resolver=None,
                             on_progress=None) -> list[str]:
    """Render every tensor-only hook anchor to disk and annotate the hooks in place.

    Each tapped anchor gains a ``tapped`` list of absolute paths, which
    :func:`src.utils.canvas_hooks.describe_hooks` renders in the ``[CANVAS HOOKS]``
    block; the flat list of new paths is returned so the caller can also hand them
    to a vision-capable orchestrator as image blocks.

    Best-effort throughout: a ComfyUI that is down, a graph that fails to execute,
    or a timeout leaves the hooks exactly as they were — the turn still runs, the
    agent just doesn't get to see the wire.
    """
    if not taps_enabled() or not isinstance(prompt, dict) or not prompt:
        return []
    taps = plan_taps(hooks, prompt, resolver=resolver)
    if not taps:
        return []
    tap_prompt, sinks = build_tap_prompt(prompt, taps)
    if not sinks:
        return []

    def _say(text: str) -> None:
        if on_progress is not None:
            try:
                on_progress(text)
            except Exception:  # noqa: BLE001
                pass

    what = ", ".join(f"node {t.node_id} ({t.out_type})" for t in sinks.values())
    _say(f"🔎 Rendering hook input(s) so I can see them: {what} …")
    timeout = float(_env_int("AGENTY_HOOK_TAP_TIMEOUT", 300))
    dest_dir = Path(tempfile.mkdtemp(prefix="agenty_tap_"))
    try:
        entry = _await_history(_submit(tap_prompt), timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("canvas_tap: tap run failed: %s", exc)
        _say(f"⚠️ Couldn't render the hook input(s) ({exc}) — describing the node(s) instead.")
        return []

    produced = _collect(entry, sinks, dest_dir)
    paths: list[str] = []
    for sink_id, files in produced.items():
        tap = sinks[sink_id]
        for anchor in tap.anchors:  # every hook reading this wire sees the same files
            anchor["tapped"] = list(files)
            anchor["tapped_type"] = tap.out_type
        paths.extend(files)
    if not paths:
        _say("⚠️ The hook input(s) rendered but produced no fetchable file.")
        return []
    _say(f"🖼️ Captured {len(paths)} file(s) from the wired hook input(s).")
    return paths
