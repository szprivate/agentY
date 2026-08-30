"""How much an output looks like the reference it was supposed to match.

The written half of a QA briefing asks the model *"is this the same character"*
and gets an opinion. There is a measurement for that question, and it is much
better than an opinion: a face embedding, compared by cosine, tells you whether
two pictures show the same person on a scale that means something.

Two scorers, for two different questions:

* :func:`face_match` — ArcFace (InsightFace). Answers *"same person?"*. On this
  machine's own renders it puts the same character at 0.95-0.98 and different
  characters below 0.54, including on heavily stylised faces, which is the case
  a photographic model is not obliged to handle.
* :func:`subject_match` — DreamSim, which was trained on human judgements of
  *diffusion-generated* image triplets, so its notion of "alike" is fitted to
  exactly the pictures agentY makes. Answers *"does this look like the
  reference"* for anything that is not a face: a location, a product, a grade.

Both are optional. Neither is imported until something asks for it, because
DreamSim's first load is ~100 s and 3 GB, and a QA pass that does not compare
against a reference must not pay any of that. Both are cached per process, and
both default to the CPU: the GPU belongs to ComfyUI, and a QA pass that fights
the render it is judging is a bad trade at any accuracy.

Weights live under ``<agentY>/models`` (gitignored) rather than in a home
directory — they are large, they belong to this checkout, and the drive they
land on should be a decision rather than a default.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("agentY.likeness")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Cosine similarity of two ArcFace embeddings. Measured on this machine's own
# output: the same character across renders scores 0.95-0.98, different
# characters 0.09-0.54. Published thresholds put 1-in-10k false accepts near
# 0.36 and strict verification at 0.65-0.75, which the observed gap agrees with.
FACE_BANDS = ((0.30, "different person"), (0.50, "possibly the same"),
              (0.70, "likely the same person"))

# DreamSim returns a DISTANCE: 0 is identical. The same scene twice measured
# 0.001 here; unrelated subjects 0.38-0.87.
SUBJECT_BANDS = ((0.10, "the same subject"), (0.35, "clearly related"),
                 (0.60, "loosely similar"))

_face_app = None
_dreamsim = None


def _band(value: float, bands, above: str) -> str:
    for edge, name in bands:
        if value < edge:
            return name
    return above


def _face_analyser():
    """The ArcFace pipeline, built once. ``None`` when it cannot be loaded."""
    global _face_app
    if _face_app is not None:
        return _face_app or None
    # Keep the download beside the checkout rather than in the user's profile.
    os.environ.setdefault("INSIGHTFACE_HOME", str(MODELS_DIR / "insightface"))
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = app
    except Exception as exc:  # noqa: BLE001
        logger.info("likeness: face matching unavailable — %s", exc)
        _face_app = False
        return None
    return _face_app


def _read_bgr(path: str):
    """An image as OpenCV's BGR array, via numpy so a non-ASCII path works.

    ``cv2.imread`` returns None for those without saying why, and most paths on
    this machine have one.
    """
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        logger.debug("likeness: could not read %s — %s", path, exc)
        return None


def _largest_face(app, path: str):
    """The biggest face in *path* — the subject, not a bystander."""
    img = _read_bgr(path)
    if img is None:
        return None
    try:
        faces = app.get(img)
    except Exception as exc:  # noqa: BLE001
        logger.debug("likeness: detection failed on %s — %s", path, exc)
        return None
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _candidates(output) -> list[str]:
    """*output* as a list of paths.

    A still is one picture; a clip is the frames sampled from it, and a character
    who is only in shot for part of it must still count as matched. So several
    candidates are allowed and the best one wins.
    """
    if isinstance(output, (str, Path)):
        return [str(output)] if str(output).strip() else []
    return [str(o) for o in (output or []) if o]


def face_match(output, references) -> dict:
    """Best face similarity between *output* and any of *references*.

    ``{}`` when the question cannot be asked — no model, no face in the output, or
    no face in any reference. That is not a failure and must never be reported as
    one: a landscape has no face, and neither does a briefing that compares one.
    """
    refs = [str(r) for r in (references or []) if r]
    outs = _candidates(output)
    if not refs or not outs:
        return {}
    app = _face_analyser()
    if app is None:
        return {}
    out_faces = [f for f in (_largest_face(app, o) for o in outs) if f is not None]
    if not out_faces:
        return {"available": False, "why": "no face detected in the output"}

    import numpy as np
    # Reference faces are detected once, not once per candidate: detection is the
    # expensive half, and a clip is compared frame by frame against the same refs.
    ref_faces = [(r, f) for r, f in ((r, _largest_face(app, r)) for r in refs)
                 if f is not None]
    if not ref_faces:
        return {"available": False, "why": "no face detected in any reference image"}
    best, best_ref = None, ""
    for ref, ref_face in ref_faces:
        for out_face in out_faces:
            score = float(np.dot(out_face.normed_embedding, ref_face.normed_embedding))
            if best is None or score > best:
                best, best_ref = score, ref
    return {
        "available": True,
        "score": round(best, 3),
        "band": _band(best, FACE_BANDS, "the same person"),
        "reference": Path(best_ref).name,
        "compared": len(ref_faces),
    }


def _dreamsim_model():
    """DreamSim, built once. ``None`` when it cannot be loaded."""
    global _dreamsim
    if _dreamsim is not None:
        return _dreamsim or None
    try:
        from dreamsim import dreamsim
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model, preprocess = dreamsim(pretrained=True, device="cpu",
                                     cache_dir=str(MODELS_DIR))
        _dreamsim = (model, preprocess)
    except Exception as exc:  # noqa: BLE001
        logger.info("likeness: subject matching unavailable — %s", exc)
        _dreamsim = False
        return None
    return _dreamsim


def subject_match(output, references) -> dict:
    """Best perceptual similarity between *output* and any of *references*.

    For everything a face embedding cannot answer — a location, a product, a
    grade. Reported as a similarity (1 = identical) rather than DreamSim's own
    distance, so it reads the same way round as the face score next to it.
    """
    refs = [str(r) for r in (references or []) if r]
    outs = _candidates(output)
    if not refs or not outs:
        return {}
    loaded = _dreamsim_model()
    if loaded is None:
        return {}
    model, preprocess = loaded
    try:
        import torch
        from PIL import Image

        def prep(p):
            with Image.open(p) as im:
                return preprocess(im.convert("RGB"))

        out_ts = [prep(o) for o in outs]
        # Each reference is preprocessed once and reused across the candidates,
        # for the same reason the face scorer detects reference faces once.
        best, best_ref = None, ""
        with torch.no_grad():
            for ref in refs:
                try:
                    ref_t = prep(ref)
                    dist = min(float(model(o, ref_t)) for o in out_ts)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("likeness: could not compare %s — %s", ref, exc)
                    continue
                if best is None or dist < best:
                    best, best_ref = dist, ref
    except Exception as exc:  # noqa: BLE001
        logger.info("likeness: subject comparison failed — %s", exc)
        return {}
    if best is None:
        return {}
    return {
        "available": True,
        "distance": round(best, 4),
        "score": round(1.0 - min(best, 1.0), 3),
        "band": _band(best, SUBJECT_BANDS, "a different subject"),
        "reference": Path(best_ref).name,
        "compared": len(refs),
    }


def render_likeness(facts: dict) -> list[str]:
    """The similarity scores as lines for the QA agent's facts block."""
    lines: list[str] = []
    face = (facts or {}).get("face_match") or {}
    if face.get("available"):
        lines.append(f"- face match: {face['score']} vs {face['reference']} "
                     f"({face['band']}; cosine, 1.0 is identical)")
    elif face.get("why"):
        lines.append(f"- face match: not measurable — {face['why']}")

    subject = (facts or {}).get("subject_match") or {}
    if subject.get("available"):
        lines.append(f"- subject match: {subject['score']} vs "
                     f"{subject['reference']} ({subject['band']}; perceptual "
                     "similarity, 1.0 is identical)")
    return lines
