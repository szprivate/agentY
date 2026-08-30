"""Measured technical properties of a finished image or video frame.

The QA judge is a vision model, and vision models are reliably bad at exactly the
properties that are cheap to compute. :func:`src.utils.qa.measure_output` already
exploits that for dimensions and duration; this is the same trade applied to the
things people actually complain about — *is it soft, is it noisy, is it blown
out*. The numbers are computed here and handed to the judge as facts, so it can
reason about the briefing instead of squinting.

Three rules shape the whole module.

**Report, do not rule.** Nothing here decides pass or fail. A soft image is
correct for a briefing that asked for a shallow depth of field and wrong for one
that asked for a product shot, and only the briefing knows which. So every number
comes with a plain-language band and no verdict.

**Scale first, or the numbers mean nothing.** Sharpness and noise measures scale
with resolution: the same picture at 512px and at 4096px gives wildly different
readings, so a threshold tuned on one is nonsense on the other. Everything is
measured on the image resized to a fixed working size, which makes a 4K render
and a thumbnail comparable — and makes the bands below transferable.

**Local, not just global.** A single blur number cannot tell a badly-rendered
frame from a portrait with a deliberately soft background; both are "blurry" on
average. Measuring the sharpest region separately answers that, and it is the
distinction a QA note has to get right to be worth reading.

Deliberately dependency-free beyond what agentY already installs (numpy, OpenCV,
Pillow). No model, no GPU, no network — this runs while ComfyUI is busy.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("agentY.image_facts")

# Everything is measured at this size (longest side), so a 4K render and a 512px
# thumbnail produce comparable numbers and one set of bands fits both.
WORK_PX = 1024

# Bands for the sharpness score at WORK_PX, read off real agentY output rather
# than picked from a blog post: across 87 renders and reference photos the score
# runs 10 to 457, quartiles 40 / 81 / 136. A deliberate Gaussian blur lands near
# 1 — below everything real — which is what makes "very soft" mean something
# rather than being the bottom of the normal range.
SHARP_BANDS = ((15.0, "very soft"), (45.0, "soft"), (150.0, "sharp"))

# Estimated noise sigma, 0-255 scale, on the same working size.
NOISE_BANDS = ((1.5, "clean"), (4.0, "light grain"), (9.0, "grainy"))

# Fraction of pixels pinned at the ends of the range before it is worth saying.
CLIP_NOTABLE = 0.02


def _band(value: float, bands) -> str:
    """The first band *value* falls under, or a name for 'above them all'."""
    for edge, name in bands:
        if value < edge:
            return name
    return {"sharp": "very sharp", "grainy": "very grainy"}.get(bands[-1][1],
                                                                bands[-1][1])


def _load_gray(path: str):
    """The image as a working-size greyscale array, or None.

    Read through Pillow rather than OpenCV's imread: OpenCV silently returns None
    for a path with a non-ASCII character, which on this machine is most of them.
    """
    import numpy as np
    from PIL import Image
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            longest = max(im.width, im.height)
            if longest > WORK_PX:
                scale = WORK_PX / longest
                im = im.resize((max(1, round(im.width * scale)),
                                max(1, round(im.height * scale))),
                               Image.BILINEAR)
            return np.asarray(im, dtype="uint8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("image_facts: could not read %s — %s", path, exc)
        return None


def sharpness(gray) -> dict:
    """How sharp the frame is, globally and at its sharpest point.

    Variance of the Laplacian, which rises with edge energy. The `sharpest_region`
    half is what separates "this render came out soft" from "the background is
    meant to be soft": a shallow depth of field has a low global score and a high
    local one, and calling that a defect would be wrong.
    """
    import cv2
    import numpy as np
    # Measured on a denoised copy. Raw variance-of-Laplacian counts grain as edge
    # energy, so a noisy frame scores as SHARPER than the clean original — adding
    # sigma-18 noise to a photo took it from 1394 to 4173, which would have told
    # the judge a degraded render was unusually crisp.
    #
    # The kernel is 5, chosen by measuring: on that same photo a 3px median still
    # let noise inflate the score by 22%, 5px brings it to 0.93x (noise now
    # slightly REDUCES it, which is honest), and 7px starts destroying real detail
    # (a clean 487 falls to 160). Blur is unaffected either way — a Gaussian
    # blur-6 collapses to about 1 at every kernel — so 5 costs nothing that
    # matters and fixes the failure mode.
    lap = cv2.Laplacian(cv2.medianBlur(gray, 5), cv2.CV_64F)
    overall = float(lap.var())

    # Tile the frame and take a high percentile, not the max: one hot pixel or a
    # compression block should not be able to declare the image sharp.
    h, w = gray.shape
    ty, tx = max(1, h // 8), max(1, w // 8)
    tiles = [lap[y:y + ty, x:x + tx].var()
             for y in range(0, h - ty + 1, ty)
             for x in range(0, w - tx + 1, tx)]
    best = float(np.percentile(tiles, 90)) if tiles else overall
    return {
        "score": round(overall, 1),
        "band": _band(overall, SHARP_BANDS),
        "sharpest_region": round(best, 1),
        "sharpest_band": _band(best, SHARP_BANDS),
    }


def noise(gray) -> dict:
    """Estimated noise/grain, as a sigma on the 0-255 scale.

    The residual after a median filter is mostly grain and fine detail. Taking a
    low percentile of its local spread keeps texture and edges — which are also
    high-frequency — from being counted as noise.
    """
    import cv2
    import numpy as np
    smoothed = cv2.medianBlur(gray, 3)
    residual = gray.astype("float32") - smoothed.astype("float32")
    h, w = residual.shape
    ty, tx = max(1, h // 8), max(1, w // 8)
    spreads = [residual[y:y + ty, x:x + tx].std()
               for y in range(0, h - ty + 1, ty)
               for x in range(0, w - tx + 1, tx)]
    # The flattest areas carry the grain without the detail.
    sigma = float(np.percentile(spreads, 25)) if spreads else float(residual.std())
    return {"sigma": round(sigma, 2), "band": _band(sigma, NOISE_BANDS)}


def exposure(gray) -> dict:
    """Brightness, contrast, and how much of the frame is pinned at either end.

    Clipping is the measurable half of "blown out" and "crushed": pixels at 0 or
    255 hold no detail and none can be recovered, so a large fraction is a fact
    worth putting in front of the judge.
    """
    import numpy as np
    arr = gray.astype("float32")
    total = arr.size or 1
    black = float((gray <= 2).sum()) / total
    white = float((gray >= 253).sum()) / total
    return {
        "mean": round(float(arr.mean()), 1),
        "contrast": round(float(arr.std()), 1),
        "clipped_black": round(black, 4),
        "clipped_white": round(white, 4),
    }


def image_quality(path: str) -> dict:
    """Every measured property of one still image. ``{}`` when it cannot be read."""
    gray = _load_gray(path)
    if gray is None:
        return {}
    try:
        return {"sharpness": sharpness(gray), "noise": noise(gray),
                "exposure": exposure(gray)}
    except Exception as exc:  # noqa: BLE001 — a QA pass must survive a bad frame
        logger.debug("image_facts: could not measure %s — %s", path, exc)
        return {}


def video_quality(path: str, samples: int = 9) -> dict:
    """The same properties across a clip, plus two faults only video has.

    A frozen stretch and a black frame are both invisible in a still and obvious
    to anyone watching, and both are cheap to count. Sharpness is reported as the
    median across frames so one motion-blurred frame does not characterise the
    whole clip.
    """
    import cv2
    import numpy as np
    try:
        cap = cv2.VideoCapture(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("image_facts: could not open %s — %s", path, exc)
        return {}
    try:
        if not cap.isOpened():
            return {}
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            return {}
        picks = [int(i * (count - 1) / max(1, samples - 1)) for i in range(samples)]
        frames, sharps, noises, blacks = [], [], [], 0
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            longest = max(gray.shape)
            if longest > WORK_PX:
                s = WORK_PX / longest
                gray = cv2.resize(gray, (max(1, round(gray.shape[1] * s)),
                                         max(1, round(gray.shape[0] * s))))
            frames.append(gray)
            sharps.append(sharpness(gray)["score"])
            noises.append(noise(gray)["sigma"])
            if float(gray.mean()) < 6:
                blacks += 1
    finally:
        cap.release()

    if not frames:
        return {}
    # Consecutive sampled frames that are all but identical: the clip is not
    # moving there. Compared on the sampled frames, so this catches a stall of
    # roughly a ninth of the clip or longer, not a single duplicated frame.
    frozen = sum(1 for a, b in zip(frames, frames[1:])
                 if float(np.abs(a.astype("float32") - b.astype("float32")).mean()) < 1.0)
    med = float(np.median(sharps))
    return {
        "sharpness": {"score": round(med, 1), "band": _band(med, SHARP_BANDS),
                      "frame_spread": round(float(np.std(sharps)), 1)},
        "noise": {"sigma": round(float(np.median(noises)), 2),
                  "band": _band(float(np.median(noises)), NOISE_BANDS)},
        "frames_sampled": len(frames),
        "black_frames": blacks,
        "frozen_pairs": frozen,
    }


def render_quality(facts: dict) -> list[str]:
    """The measured properties as lines for the QA agent's facts block.

    Every line carries the number AND the band. The number is what a criterion
    with a threshold is checked against; the band is so a judge that was given no
    such criterion still reads the line correctly instead of inventing a scale.
    """
    if not facts:
        return []
    lines: list[str] = []

    sharp = facts.get("sharpness") or {}
    if sharp:
        line = f"- sharpness: {sharp['score']} ({sharp['band']})"
        # The shallow depth-of-field signature, and ONLY that: a soft frame whose
        # sharpest part is genuinely sharp. Firing whenever the two bands merely
        # differ claimed "parts of the frame are sharp" about a hazy render whose
        # best region scored 52 — soft as well, just less so.
        soft_overall = sharp["band"] in ("very soft", "soft")
        crisp_somewhere = sharp.get("sharpest_band") in ("sharp", "very sharp")
        if soft_overall and crisp_somewhere:
            line += (f"; sharpest region {sharp['sharpest_region']} "
                     f"({sharp['sharpest_band']}) — part of the frame IS sharp, so "
                     "this may be depth of field rather than a soft render")
        if sharp.get("frame_spread") is not None:
            line += f"; spread across frames {sharp['frame_spread']}"
        lines.append(line)

    ns = facts.get("noise") or {}
    if ns:
        lines.append(f"- noise/grain: sigma {ns['sigma']} ({ns['band']})")

    ex = facts.get("exposure") or {}
    if ex:
        lines.append(f"- exposure: mean {ex['mean']}/255, contrast {ex['contrast']}")
        pins = []
        if ex.get("clipped_black", 0) >= CLIP_NOTABLE:
            pins.append(f"{ex['clipped_black'] * 100:.1f}% crushed to black")
        if ex.get("clipped_white", 0) >= CLIP_NOTABLE:
            pins.append(f"{ex['clipped_white'] * 100:.1f}% blown to white")
        if pins:
            lines.append("- clipping: " + ", ".join(pins) + " (no detail there to recover)")

    if facts.get("black_frames"):
        lines.append(f"- black frames: {facts['black_frames']} of "
                     f"{facts.get('frames_sampled', '?')} sampled")
    if facts.get("frozen_pairs"):
        lines.append(f"- motion: {facts['frozen_pairs']} sampled pair(s) essentially "
                     "identical — the clip may stall there")
    return lines


def measure(path: str, *, is_video: bool = False) -> dict:
    """Measured quality facts for one output. ``{}`` when it cannot be read."""
    return video_quality(path) if is_video else image_quality(str(path))
