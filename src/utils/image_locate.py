"""Resolve "the bolts" in a picture to coordinates, so they can be marked.

This is the *locating* half of the annotation feature; the drawing half lives in
:mod:`agenty_core.utils.image_annotate` and never learns where a region came
from. Both sides speak :class:`~agenty_core.utils.image_annotate.Region`, so a
grounding model, a vision model and a user typing coordinates are
interchangeable here.

The default backend is SAM 3 running **in this process**. Going through a ComfyUI
graph instead would mean the feature is unavailable exactly when someone is
looking at a still image with ComfyUI shut down, and it would cost a queue
round-trip for something that takes 0.2s.

Two things about the weights, both of which cost an afternoon to discover:

* The checkpoint reused here is ComfyUI's own ``models/sam3/sam3.safetensors``
  (the ungated ``apozz/sam3-safetensors`` mirror). Nothing is downloaded, and the
  gated ``facebook/sam3`` repo is not needed.
* That file is the unified *video* model. The image model sits under a
  ``detector.`` key prefix, and upstream's loader is ``torch.load`` — a ``.pt``
  pickle — where this is safetensors. So the state dict is read and re-prefixed
  here rather than handed to ``build_sam3_image_model(checkpoint_path=...)``.

The model is loaded on first use and dropped again after an idle period, because
ComfyUI's ``model_management`` cannot see VRAM held by this process and therefore
cannot evict it to make room for a video model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from agenty_core.utils.image_annotate import Region, dedupe_regions

# Model id prefix in the unified checkpoint that holds the image model.
_DETECTOR_PREFIX = "detector."
# How long the model may sit unused before its VRAM is handed back.
_DEFAULT_IDLE_UNLOAD_S = 180.0
_UNLOAD_POLL_S = 20.0

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {
    "model": None,
    "processor": None,
    "device": "",
    "last_used": 0.0,
    "reaper": None,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def _settings() -> dict:
    try:
        from src.utils.settings import load_settings
        return load_settings() or {}
    except Exception:  # noqa: BLE001 — locating must not depend on settings loading
        return {}


def _cfg(env_var: str, key: str, default: Any) -> Any:
    """env var > ``[annotate]`` in settings > hard default."""
    raw = os.environ.get(env_var)
    if raw is not None and str(raw).strip() != "":
        return raw
    node = _settings().get("annotate", {})
    if isinstance(node, dict):
        val = node.get(key)
        if val not in (None, ""):
            return val
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def idle_unload_seconds() -> float:
    """0 or less disables the reaper and keeps the model resident."""
    return _as_float(_cfg("AGENTY_SAM3_IDLE_UNLOAD", "sam3_idle_unload_s",
                          _DEFAULT_IDLE_UNLOAD_S), _DEFAULT_IDLE_UNLOAD_S)


def _device() -> str:
    want = str(_cfg("AGENTY_SAM3_DEVICE", "sam3_device", "auto")).strip().lower()
    if want in ("cpu", "cuda"):
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _comfy_roots() -> list[str]:
    """Plausible ComfyUI install directories, best first.

    No absolute path is hard-coded: an install this code has never been told
    about is found either by asking a running ComfyUI or by the sibling-checkout
    convention the rest of agentY already assumes.
    """
    roots: list[str] = []
    configured = str(_settings().get("comfyui_dir", "") or "").strip()
    if configured:
        roots.append(configured)
    here = Path(os.getcwd()).resolve()
    for base in (here, *here.parents[:2]):
        roots.append(str(base.parent / "comfyui"))
        roots.append(str(base / "comfyui"))
    return [r for r in roots if r]


def _comfy_models_dir() -> Optional[str]:
    """ComfyUI's models directory, from settings, a live server, or a sibling checkout."""
    configured = str(_settings().get("comfyui_models_dir", "") or "").strip()
    if configured and os.path.isdir(configured):
        return configured
    try:
        from src.tools.comfyui import get_comfyui_dirs
        info = json.loads(get_comfyui_dirs()) or {}
        for key in ("models_dir", "model_dir"):
            v = info.get(key)
            if v and v != "unknown" and os.path.isdir(v):
                return v
        base = info.get("base_dir") or info.get("comfyui_dir")
        if base and os.path.isdir(os.path.join(base, "models")):
            return os.path.join(base, "models")
    except Exception:  # noqa: BLE001 — ComfyUI may simply be down; that is fine
        pass
    for root in _comfy_roots():
        cand = os.path.join(root, "models")
        if os.path.isdir(cand):
            return cand
    return None


def checkpoint_path() -> Optional[str]:
    """Locate ``sam3.safetensors``: explicit setting, then ComfyUI's models dir."""
    explicit = str(_cfg("AGENTY_SAM3_CHECKPOINT", "sam3_checkpoint", "") or "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    models = _comfy_models_dir()
    if models:
        cand = os.path.join(models, "sam3", "sam3.safetensors")
        if os.path.isfile(cand):
            return cand
    return None


def bpe_path() -> Optional[str]:
    """Locate the CLIP BPE vocabulary SAM3's text encoder tokenises with.

    Upstream's default points at an ``assets/`` folder that only exists in a repo
    checkout, not in the wheel — so it has to be found. The ComfyUI-SAM3 node
    ships a copy, which is the usual place it turns up on a machine that already
    runs SAM3 there.
    """
    explicit = str(_cfg("AGENTY_SAM3_BPE", "sam3_bpe", "") or "").strip()
    if explicit:
        return explicit if os.path.isfile(explicit) else None

    name = "bpe_simple_vocab_16e6.txt.gz"
    candidates: list[str] = []
    try:
        import sam3 as _sam3
        pkg = Path(_sam3.__file__).parent
        candidates += [str(pkg / "assets" / name), str(pkg.parent / "assets" / name)]
    except Exception:  # noqa: BLE001
        pass
    models = _comfy_models_dir()
    roots = list(_comfy_roots())
    if models:
        roots.insert(0, str(Path(models).parent))
    for root in roots:
        candidates.append(os.path.join(root, "custom_nodes", "ComfyUI-SAM3",
                                       "nodes", "sam3", name))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def availability() -> tuple[bool, str]:
    """``(usable, reason)`` — why the SAM3 backend cannot run, in plain words."""
    try:
        import sam3  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"the sam3 package is not importable ({exc}). Install it with "
            "`pip install -r requirements.txt`; note torch must come from the "
            "CUDA index first or it will be CPU-only."
        )
    if checkpoint_path() is None:
        return False, (
            "sam3.safetensors was not found. Expected it in ComfyUI's "
            "models/sam3/ (the ComfyUI-SAM3 node downloads it), or set "
            "annotate.sam3_checkpoint / AGENTY_SAM3_CHECKPOINT."
        )
    if bpe_path() is None:
        return False, (
            "the CLIP BPE vocabulary (bpe_simple_vocab_16e6.txt.gz) was not "
            "found. It ships with the ComfyUI-SAM3 node; set annotate.sam3_bpe "
            "or AGENTY_SAM3_BPE to point at a copy."
        )
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Model lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

def _reaper_loop() -> None:
    """Hand VRAM back once the model has gone unused for long enough."""
    while True:
        time.sleep(_UNLOAD_POLL_S)
        timeout = idle_unload_seconds()
        with _LOCK:
            if _STATE["model"] is None:
                _STATE["reaper"] = None
                return
            if timeout <= 0:
                continue
            idle = time.monotonic() - float(_STATE["last_used"] or 0.0)
            if idle < timeout:
                continue
        unload(reason=f"idle for {idle:.0f}s")


def _start_reaper() -> None:
    if _STATE.get("reaper") is not None or idle_unload_seconds() <= 0:
        return
    t = threading.Thread(target=_reaper_loop, name="sam3-idle-unload", daemon=True)
    _STATE["reaper"] = t
    t.start()


def unload(reason: str = "") -> bool:
    """Drop the model and release its VRAM. Returns True if something was freed."""
    with _LOCK:
        if _STATE["model"] is None:
            return False
        _STATE["model"] = None
        _STATE["processor"] = None
        _STATE["device"] = ""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print(f"[locate] SAM3 unloaded{f' ({reason})' if reason else ''}")
    return True


def is_loaded() -> bool:
    with _LOCK:
        return _STATE["model"] is not None


def _load() -> tuple[Any, Any]:
    """Build the image model and load ComfyUI's checkpoint into it."""
    import torch
    from safetensors.torch import load_file
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    ckpt, bpe, device = checkpoint_path(), bpe_path(), _device()
    t0 = time.monotonic()
    model = build_sam3_image_model(
        bpe_path=bpe, device=device, eval_mode=True,
        checkpoint_path=None, load_from_HF=False,
    )
    raw = load_file(ckpt)
    weights = {k[len(_DETECTOR_PREFIX):]: v
               for k, v in raw.items() if k.startswith(_DETECTOR_PREFIX)}
    if not weights:
        # A checkpoint already in image-model layout: use it as-is.
        weights = raw
    missing, _unexpected = model.load_state_dict(weights, strict=False)
    del raw, weights
    if missing:
        raise RuntimeError(
            f"SAM3 checkpoint {ckpt} is missing {len(missing)} model keys "
            f"(first: {missing[:3]}); it does not match this sam3 version."
        )
    model = model.to(device).eval()
    processor = Sam3Processor(model, device=device)
    held = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
    print(f"[locate] SAM3 loaded in {time.monotonic()-t0:.1f}s on {device}"
          + (f", holding {held:.2f} GB VRAM" if held else ""))
    return model, processor


def _ensure_loaded() -> Any:
    with _LOCK:
        if _STATE["model"] is None:
            ok, reason = availability()
            if not ok:
                raise RuntimeError(reason)
            model, processor = _load()
            _STATE["model"], _STATE["processor"] = model, processor
            _STATE["device"] = _device()
            _start_reaper()
        _STATE["last_used"] = time.monotonic()
        return _STATE["processor"]


# ═══════════════════════════════════════════════════════════════════════════════
# Locating
# ═══════════════════════════════════════════════════════════════════════════════

def _to_numpy(x: Any):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else x


def locate_sam3(
    image,
    prompt: str,
    threshold: float = 0.2,
    max_results: int = 8,
    with_masks: bool = True,
) -> list[Region]:
    """Ground *prompt* in *image* (a PIL image) and return regions in its pixels.

    ``threshold`` defaults to 0.2, not 0.5: SAM3 scores presence rather than
    classification confidence, and 0.5 routinely returns nothing at all for an
    object plainly in the frame.
    """
    import numpy as np

    processor = _ensure_loaded()
    with _LOCK:
        # set_image caches into the state dict it returns, but the model itself is
        # shared, so one call at a time.
        processor.set_confidence_threshold(float(threshold))
        # PIL, not ndarray: set_image reads shape[-2:] for arrays (i.e. it expects
        # CHW) while handing them to a HWC converter, so an ordinary HxWx3 array
        # silently yields width=3 and zero detections.
        state = processor.set_image(image.convert("RGB"))
        out = processor.set_text_prompt(prompt, state)
        _STATE["last_used"] = time.monotonic()

    boxes = _to_numpy(out.get("boxes"))
    scores = _to_numpy(out.get("scores"))
    if boxes is None or len(boxes) == 0:
        return []

    order = np.argsort(-np.asarray(scores)) if scores is not None else range(len(boxes))
    keep = list(order)[: max(1, int(max_results))]

    masks = _to_numpy(out.get("masks")) if with_masks else None
    regions: list[Region] = []
    for i in keep:
        mask = None
        if masks is not None and i < len(masks):
            m = masks[i]
            mask = m[0] if getattr(m, "ndim", 0) == 3 else m
        regions.append(Region(
            box=[float(v) for v in boxes[i]],
            label=prompt,
            score=float(scores[i]) if scores is not None and i < len(scores) else None,
            mask=mask,
        ))
    return regions


def locate(
    image,
    prompts: Sequence[str] | str,
    threshold: float = 0.2,
    max_results: int = 8,
    with_masks: bool = True,
    dedupe: bool = True,
) -> list[Region]:
    """Ground one or several prompts and return the merged, de-duplicated regions.

    Several prompts are useful because SAM3's vocabulary is uneven — "seaweed"
    scores nothing on an image where "kelp" scores well — so the caller can offer
    synonyms and take whatever lands.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    found: list[Region] = []
    for prompt in prompts:
        prompt = (prompt or "").strip()
        if not prompt:
            continue
        found.extend(locate_sam3(image, prompt, threshold, max_results, with_masks))
    if dedupe:
        found = dedupe_regions(found)
    found.sort(key=lambda r: -(r.score if r.score is not None else 0.0))
    return found[: max(1, int(max_results))]
