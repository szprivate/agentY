"""``annotate_image`` — mark up a picture the user gave us.

A fat tool on purpose: one call goes from "circle the bolts" to a finished PNG.
The model supplies language (which nouns to look for, what the marks should look
like) and nothing mechanical — locating, de-duplicating, scaling, drawing and
staging are all deterministic here. That split is what keeps it to a single tool
call instead of a conversation.

The original image is never altered. Marks are composited as an overlay, so what
comes back is the user's own photograph with ink on top, not a re-generated
lookalike.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
from strands import tool

from agenty_core.utils.image_annotate import Region, Style, annotate, dedupe_regions

# Set by the pipeline so a finished annotation reaches the chat panel and the
# canvas. Mirrors the set_vision_agent / set_video_agent injection: the tool stays
# importable and testable without a running pipeline.
_output_sink: Optional[Callable[[str], None]] = None


def set_output_sink(fn: Optional[Callable[[str], None]]) -> None:
    """Register the callable that publishes a produced file as a turn output."""
    global _output_sink
    _output_sink = fn


def _publish(path: str) -> bool:
    if _output_sink is None:
        return False
    try:
        _output_sink(path)
        return True
    except Exception as exc:  # noqa: BLE001 — never fail the tool over delivery
        print(f"[annotate_image] could not publish {path}: {exc}")
        return False


def _split_targets(raw: Any) -> list[str]:
    """Accept a JSON array, a comma-separated string, or a single phrase."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            pass
    # Semicolons first: "a, b; c" most likely means two targets, not three.
    parts = text.split(";") if ";" in text else text.split(",")
    return [p.strip() for p in parts if p.strip()]


def _parse_regions(raw: Any, width: int, height: int) -> list[Region]:
    """Explicit coordinates from the caller — the path that costs nothing.

    Boxes may be pixels or 0-1 fractions; anything with every value <= 1 is read
    as fractional, since a 1x1-pixel box is never what someone means.
    """
    if not raw:
        return []
    data = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"regions is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("regions must be a JSON array of {box, label} objects")

    out: list[Region] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            box, label = list(item), ""
        elif isinstance(item, dict):
            box = item.get("box") or item.get("bbox") or item.get("bbox_2d")
            label = str(item.get("label", "") or "")
        else:
            raise ValueError(f"unusable region entry: {item!r}")
        if not box or len(box) < 4:
            raise ValueError(f"region needs a 4-number box, got {box!r}")
        vals = [float(v) for v in list(box)[:4]]
        if all(0.0 <= v <= 1.0 for v in vals):
            vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
        out.append(Region(box=vals, label=label))
    return out


def _resolve(file_path: str) -> Optional[str]:
    """Resolve a path, including a bare LoadImage filename, with ComfyUI down.

    The shared resolver finds ComfyUI's input dir by asking the server, so a bare
    ``photo.jpg`` stops resolving the moment ComfyUI is off — which is precisely
    the situation this feature exists to survive. Fall back to the same
    sibling-checkout convention the SAM3 weights are found by.
    """
    from src.tools.image_handling import _resolve_local_image
    found = _resolve_local_image(file_path)
    if found:
        return found
    try:
        from src.utils.image_locate import _comfy_roots
        name = os.path.basename(file_path)
        if name:
            for root in _comfy_roots():
                cand = os.path.join(root, "input", name)
                if os.path.isfile(cand):
                    return cand
    except Exception:  # noqa: BLE001
        pass
    return None


def _clamp_box(box, width: int, height: int) -> list[float]:
    """Grounding returns boxes a pixel or two outside the frame; report them inside."""
    x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
    return [max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))]


def _output_dir() -> Optional[str]:
    try:
        from src.tools.comfyui import get_agent_output_dirs
        info = json.loads(get_agent_output_dirs()) or {}
        for key in ("images_dir", "image_dir", "images"):
            v = info.get(key)
            if v and v != "unknown":
                return v
    except Exception:  # noqa: BLE001 — fall back to sitting beside the source
        pass
    return None


@tool
def annotate_image(
    file_path: str,
    targets: str = "",
    regions: str = "",
    shape: str = "ellipse",
    color: str = "red",
    label: str = "none",
    threshold: float = 0.2,
    max_marks: int = 8,
    out_path: str = "",
) -> dict:
    """Draw markers on an image to point at things — circle them, box them, arrow them.

    Use this when the user asks you to MARK UP a picture: "circle the bolts",
    "put a red box around the logo", "show me where the damage is". The marks are
    composited onto the untouched original, so the photo itself is unchanged —
    never send an image to an edit/generation workflow just to draw a circle on
    it, which repaints every pixel and lands the circle in the wrong place.

    Give it EITHER ``targets`` (things to find) or ``regions`` (coordinates you
    already know). ``regions`` costs nothing; ``targets`` runs a local grounding
    model on the GPU.

    Args:
        file_path: The image to mark. A bare filename from a LoadImage widget
            works — ComfyUI's input folder is searched.
        targets: What to mark, as short concrete noun phrases separated by commas
            (``"bolt, seaweed"``) or a JSON array. Each is located independently
            and the results merged. Keep them literal and visual — "bolt" or "red
            car", not "the thing that looks broken". If one returns nothing, try
            a synonym: the grounding vocabulary is uneven, and e.g. "kelp" can hit
            where "seaweed" misses.
        regions: Explicit boxes instead of searching, as a JSON array of
            ``{"box": [x1, y1, x2, y2], "label": "..."}``. Values may be pixels or
            0-1 fractions of the image. Use this when the user names a location
            outright ("the top-left corner") — no model runs.
        shape: ``ellipse`` (default), ``rect``, ``rounded_rect``, ``arrow``,
            ``polygon`` (traces the object's real outline, needs ``targets``), or
            ``spotlight`` (dims everything else).
        color: Mark colour — a name (``red``, ``blue``, ``green``, ``yellow``) or
            ``#rrggbb``. Defaults to red.
        label: ``none`` (default), ``number`` (1, 2, 3 badges) or ``text`` (the
            target name and match confidence).
        threshold: Match confidence floor, 0-1. Default 0.2, which is right for
            this model — it scores presence rather than certainty, and 0.5 often
            returns nothing for an object plainly in frame. Raise it if you get
            marks on the wrong things.
        max_marks: Cap on how many things get marked (default 8), highest
            confidence first.
        out_path: Where to write. Defaults to a ``*_annotated.png`` next to the
            agent's other outputs.

    Returns:
        ``{"status", "output_path", "marked": [{"label", "box", "score"}], ...}``.
        The annotated image is delivered to the chat panel automatically.
    """
    resolved = _resolve(file_path)
    if resolved is None:
        return {"status": "error", "content": [{"text": f"Image not found: {file_path}"}]}

    try:
        with Image.open(resolved) as im:
            im.load()
            image = im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"Could not read {resolved}: {exc}"}]}

    width, height = image.size
    wanted = _split_targets(targets)

    # ── explicit coordinates: no model, no GPU ──────────────────────────────
    try:
        found = _parse_regions(regions, width, height)
    except ValueError as exc:
        return {"status": "error", "content": [{"text": str(exc)}]}
    located_by = "regions" if found else ""

    # ── or ground the targets ───────────────────────────────────────────────
    if not found:
        if not wanted:
            return {"status": "error", "content": [{"text": (
                "Nothing to mark: pass `targets` (e.g. targets=\"bolt, seaweed\") "
                "or `regions` with explicit coordinates."
            )}]}
        from src.utils import image_locate
        ok, why = image_locate.availability()
        if not ok:
            return {"status": "error", "content": [{"text": (
                f"Cannot locate things in the image: {why}\n\n"
                "You can still annotate by passing `regions` with explicit "
                "coordinates — call analyze_image(mode='describe') to have the "
                "vision agent say roughly where the subject is, then estimate the "
                "box from the image dimensions."
            )}]}
        try:
            found = image_locate.locate(
                image, wanted,
                threshold=float(threshold),
                max_results=max(1, int(max_marks)),
                with_masks=(str(shape).strip().lower() in ("polygon", "spotlight")),
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "content": [{"text": f"Locating failed: {exc}"}]}
        located_by = "sam3"

    if not found:
        return {"status": "error", "content": [{"text": (
            f"Nothing matching {wanted!r} was found in {os.path.basename(resolved)} "
            f"at threshold {threshold}. Try a different word for the same thing "
            f"(the grounding vocabulary is uneven), or lower the threshold."
        )}]}

    found = dedupe_regions(found)[: max(1, int(max_marks))]

    style = Style(
        color=color or "red",
        shape=shape or "ellipse",
        label_mode=(label or "none"),
    )
    try:
        marked = annotate(image, found, style)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"Drawing failed: {exc}"}]}

    src = Path(resolved)
    if out_path:
        dest = Path(out_path)
    else:
        out_dir = _output_dir()
        base = f"{src.stem}_annotated.png"
        dest = Path(out_dir) / base if out_dir else src.with_name(base)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        marked.convert("RGBA").save(dest, format="PNG")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"Could not write {dest}: {exc}"}]}

    delivered = _publish(str(dest))
    summary = ", ".join(
        f"{r.label or 'region'}"
        + (f" ({r.score:.0%})" if r.score is not None else "")
        for r in found
    )
    return {
        "status": "success",
        "output_path": str(dest),
        "source": str(src),
        "located_by": located_by,
        "shape": style.shape,
        "marked": [
            {"label": r.label,
             "box": [round(v, 1) for v in _clamp_box(r.box, width, height)],
             "score": round(float(r.score), 3) if r.score is not None else None}
            for r in found
        ],
        "content": [{"text": (
            f"Marked {len(found)} region(s) on {src.name} with {style.shape}s: {summary}.\n"
            f"Saved to {dest}."
            + ("" if delivered else "\n(Not auto-delivered to the panel; "
                                    "reference the path above.)")
        )}],
    }
