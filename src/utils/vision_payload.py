"""Image bytes as they are sent to a hosted model inside the conversation.

DashScope does not read an inline image from the request itself: it stores the
base64 data and its model worker then fetches it, and when that fetch is slow
the whole call fails with ``invalid_parameter_error`` — "Download multimodal
file timed out". Images embedded in the orchestrator's messages (attachments,
the viewer's picks, ``analyze_image(mode='full')``) were PNGs of up to ~3.6 MB
at 1568 px, several per message, re-sent on every turn, which is what made that
error common.

:func:`for_model` is the one place that prepares such bytes: downsized to the
usual limits, opaque PNG re-encoded as JPEG (as ``analyze_image`` already did for
its describe call), and for DashScope a smaller long edge. ``llm.embed_max_edge``
overrides that edge for every provider (0 = no extra cap).
"""
from __future__ import annotations

import io

# DashScope's own guidance for Qwen-VL input tops out around here; past it the
# model downsamples anyway, so the extra pixels only cost upload time.
DASHSCOPE_EMBED_EDGE = 1280

_DASHSCOPE = {"dashscope", "modelstudio", "qwen", "alibaba"}


def embed_edge(spec: str = "") -> int:
    """Long-edge cap (px) for images embedded for the model *spec*, 0 for none."""
    try:
        from src.utils.settings import load_settings
        raw = (load_settings().get("llm") or {}).get("embed_max_edge")
        if raw is not None and str(raw).strip() != "":
            return max(0, int(raw))
    except Exception:  # noqa: BLE001 - settings must never cost the image
        pass
    provider = str(spec or "").lower().partition(",")[0].strip()
    return DASHSCOPE_EMBED_EDGE if provider in _DASHSCOPE else 0


def _cap_edge(data: bytes, fmt: str, edge: int) -> tuple[bytes, str]:
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    long_edge = max(img.width, img.height)
    if not edge or long_edge <= edge:
        return data, fmt
    ratio = edge / long_edge
    img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
                     Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", optimize=True)
    elif fmt in ("jpeg", "jpg"):
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        fmt = "jpeg"
    else:                       # gif / webp: keep the bytes, they are already small
        return data, fmt
    return buf.getvalue(), fmt


def for_model(data: bytes, fmt: str, spec: str = "") -> tuple[bytes, str]:
    """*data* (format *fmt*) ready to embed for the model *spec*.

    Returns ``(bytes, format)``. Never raises over the re-encode: whatever step
    fails, the bytes from the step before are returned.
    """
    from src.tools.image_handling import _downsize, _jpeg_for_vision
    data, fmt = _downsize(data, fmt)
    if fmt == "png":
        data, fmt = _jpeg_for_vision(data)     # keeps PNG only for real transparency
    try:
        data, fmt = _cap_edge(data, fmt, embed_edge(spec))
    except Exception:  # noqa: BLE001
        pass
    return data, fmt
