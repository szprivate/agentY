"""The edit as a document.

An ``agentY edit`` stage cuts approved shots into one film. The agent does not
write ffmpeg commands for that, and does not write the film over from scratch
when the user says "shot 3 shorter": it keeps a **timeline** - which clip, from
where to where, in which order, how one passes into the next - and changes it
with small operations. The host checks the timeline and renders it
(:mod:`src.utils.edit_render`), and saves it as OpenTimelineIO, which Resolve,
Nuke Studio and Premiere open.

Pure data and arithmetic: no files are read here and nothing is run.

A timeline::

    {"fps": 24.0, "width": 1920, "height": 1080,
     "clips": [{"id": "c1", "src": "<path>", "in": 0.0, "out": 3.5,
                "transition": 0.0, "label": "kitchen wide"}, ...],
     "audio": {"src": "<path>", "in": 0.0, "gain": 1.0, "fade_out": 1.0} | None,
     "overlays": []}

Times are seconds. ``in``/``out`` are positions in the SOURCE; ``transition`` is
the length of the dissolve from the clip before into this one (0 = a cut), and
it is taken out of both, so a dissolve shortens the film by its length.
"""

from __future__ import annotations

import copy
import os

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr")
FORMATS = ("mp4", "mov")
_MIN_CLIP = 0.1            # seconds: shorter than this is a mistake, not a cut
_STILL_DEFAULT = 3.0       # how long a still stands when nothing says


class EditError(ValueError):
    """An operation that cannot be applied; the message is for the agent."""


def new(fps: float = 24.0, width: int = 1920, height: int = 1080) -> dict:
    return {"fps": float(fps or 24.0), "width": int(width or 1920), "height": int(height or 1080),
            "clips": [], "audio": None, "overlays": []}


def is_still(src: str) -> bool:
    return str(src or "").lower().endswith(IMAGE_EXTS)


def clip_length(clip: dict) -> float:
    return max(0.0, float(clip.get("out") or 0.0) - float(clip.get("in") or 0.0))


def duration(timeline: dict) -> float:
    """How long the film is: the clips, less what the dissolves take."""
    clips = timeline.get("clips") or []
    total = sum(clip_length(c) for c in clips)
    return max(0.0, total - sum(float(c.get("transition") or 0.0) for c in clips[1:]))


def starts(timeline: dict) -> list:
    """Where each clip begins in the film, in seconds."""
    at, out = 0.0, []
    for i, c in enumerate(timeline.get("clips") or []):
        if i:
            at -= float(c.get("transition") or 0.0)
        out.append(round(at, 4))
        at += clip_length(c)
    return out


def _next_id(timeline: dict) -> str:
    used = {str(c.get("id")) for c in timeline.get("clips") or []}
    n = len(used) + 1
    while f"c{n}" in used:
        n += 1
    return f"c{n}"


def _find(timeline: dict, clip_id) -> int:
    want = str(clip_id or "").strip()
    clips = timeline.get("clips") or []
    for i, c in enumerate(clips):
        if str(c.get("id")) == want:
            return i
    # "3" or 3 also means the third clip: that is how a person counts them.
    if want.isdigit() and 1 <= int(want) <= len(clips):
        return int(want) - 1
    raise EditError(f"there is no clip '{clip_id}' - the clips are "
                    + (", ".join(str(c.get("id")) for c in clips) or "none yet"))


def _seconds(value, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise EditError(f"{what} must be a number of seconds, not {value!r}") from None
    if v < 0:
        raise EditError(f"{what} cannot be negative ({v})")
    return v


def _span(op: dict, facts: dict | None, src: str, was: dict | None = None) -> tuple[float, float]:
    """The in and out an operation asks for, filled in from the source's length."""
    known = float(((facts or {}).get(src) or {}).get("duration") or 0.0)
    start = _seconds(op["in"], "in") if op.get("in") is not None else float((was or {}).get("in") or 0.0)
    if op.get("out") is not None:
        end = _seconds(op["out"], "out")
    elif op.get("length") is not None:
        end = start + _seconds(op["length"], "length")
    elif was is not None:
        end = float(was.get("out") or 0.0)
    elif is_still(src):
        end = start + _STILL_DEFAULT
    elif known:
        end = known
    else:
        raise EditError(f"say where the clip ends (out, or length): the length of "
                        f"{os.path.basename(src)} is not known")
    if end - start < _MIN_CLIP:
        raise EditError(f"a clip from {start:g}s to {end:g}s is {end - start:g}s long - too short")
    return round(start, 4), round(end, 4)


def apply(timeline: dict, ops, facts: dict | None = None) -> dict:
    """*timeline* with *ops* applied, in order. The original is not changed.

    *facts* - ``{src: {"duration": seconds}}`` for the sources that have been
    looked at; with it a clip can be added without saying where it ends.
    Raises :class:`EditError` naming the operation that cannot be done; nothing
    is applied then.
    """
    tl = copy.deepcopy(timeline) if timeline else new()
    tl.setdefault("clips", [])
    tl.setdefault("audio", None)
    tl.setdefault("overlays", [])
    for n, op in enumerate(ops if isinstance(ops, (list, tuple)) else [ops], 1):
        if not isinstance(op, dict):
            raise EditError(f"operation {n} is not an object")
        kind = str(op.get("op") or "").strip().lower()
        try:
            _apply_one(tl, kind, op, facts)
        except EditError as exc:
            raise EditError(f"operation {n} ({kind or '?'}): {exc}") from None
    return tl


def _apply_one(tl: dict, kind: str, op: dict, facts: dict | None) -> None:
    clips = tl["clips"]
    if kind == "add":
        src = str(op.get("src") or "").strip()
        if not src:
            raise EditError("give the file to add (src)")
        start, end = _span(op, facts, src)
        clip = {"id": _next_id(tl), "src": src, "in": start, "out": end,
                "transition": _seconds(op.get("transition") or 0, "transition"),
                "label": str(op.get("label") or "").strip()}
        at = op.get("at")
        if at is None:
            clips.append(clip)
        else:
            clips.insert(max(0, min(len(clips), int(at) - 1)), clip)
    elif kind == "trim":
        c = clips[_find(tl, op.get("clip"))]
        c["in"], c["out"] = _span(op, facts, c["src"], c)
    elif kind == "move":
        i = _find(tl, op.get("clip"))
        to = op.get("to")
        if to is None:
            raise EditError("say where it goes (to: its new position, 1 = first)")
        clip = clips.pop(i)
        clips.insert(max(0, min(len(clips), int(to) - 1)), clip)
    elif kind == "replace":
        c = clips[_find(tl, op.get("clip"))]
        src = str(op.get("src") or "").strip()
        if not src:
            raise EditError("give the file that takes its place (src)")
        length = clip_length(c)
        c["src"] = src
        # The new file takes the old one's place in the cut: same length, from
        # its own beginning, unless the operation says otherwise.
        fresh = dict(op)
        if fresh.get("in") is None and fresh.get("out") is None and fresh.get("length") is None:
            fresh["in"], fresh["length"] = 0.0, length
        c["in"], c["out"] = _span(fresh, facts, src)
        if op.get("label") is not None:
            c["label"] = str(op.get("label") or "").strip()
    elif kind == "remove":
        clips.pop(_find(tl, op.get("clip")))
    elif kind == "transition":
        i = _find(tl, op.get("clip"))
        if i == 0:
            raise EditError("the first clip has nothing before it to dissolve from")
        clips[i]["transition"] = _seconds(op.get("seconds") or 0, "seconds")
    elif kind == "label":
        clips[_find(tl, op.get("clip"))]["label"] = str(op.get("label") or "").strip()
    elif kind == "audio":
        src = str(op.get("src") or "").strip()
        tl["audio"] = None if not src else {
            "src": src, "in": _seconds(op.get("in") or 0, "in"),
            "gain": float(op.get("gain") if op.get("gain") is not None else 1.0),
            "fade_out": _seconds(op.get("fade_out") if op.get("fade_out") is not None else 1.0,
                                 "fade_out")}
    elif kind == "fit":
        fit(tl, _seconds(op.get("seconds"), "seconds"))
    elif kind == "settings":
        for key, cast in (("fps", float), ("width", int), ("height", int)):
            if op.get(key):
                tl[key] = cast(op[key])
    elif kind == "clear":
        tl["clips"], tl["audio"] = [], None
    else:
        raise EditError("unknown operation - use add, trim, move, replace, remove, "
                        "transition, label, audio, fit, settings or clear")


def fit(timeline: dict, seconds: float) -> dict:
    """Shorten the clips, each by the same share, so the film is *seconds* long.

    Taken off each clip's END: where a shot starts is usually what was chosen.
    A film already that short is left alone - nothing here makes one longer.
    """
    now = duration(timeline)
    if seconds <= 0 or now <= seconds + 1e-6:
        return timeline
    clips = timeline["clips"]
    lost = sum(float(c.get("transition") or 0.0) for c in clips[1:])
    share = (seconds + lost) / (now + lost)
    for c in clips:
        length = max(_MIN_CLIP, clip_length(c) * share)
        c["out"] = round(float(c["in"]) + length, 4)
    return timeline


def problems(timeline: dict, facts: dict | None = None) -> list:
    """What stops this timeline from being rendered, in words (empty = nothing)."""
    out = []
    clips = timeline.get("clips") or []
    if not clips:
        return ["the timeline has no clips yet"]
    for i, c in enumerate(clips, 1):
        name = f"clip {c.get('id')} ({os.path.basename(str(c.get('src') or ''))})"
        length = clip_length(c)
        if length < _MIN_CLIP:
            out.append(f"{name} has no length")
        fact = (facts or {}).get(c.get("src"))
        if facts is not None and fact is None:
            out.append(f"{name}: the file is missing or is not a picture or a video")
        elif fact and not is_still(c.get("src")) and fact.get("duration") \
                and float(c.get("out") or 0) > float(fact["duration"]) + 0.05:
            out.append(f"{name} ends at {float(c['out']):g}s but the file is only "
                       f"{float(fact['duration']):g}s long")
        t = float(c.get("transition") or 0.0)
        if i > 1 and t > 0:
            before = clip_length(clips[i - 2])
            if t >= length or t >= before:
                out.append(f"the {t:g}s dissolve into {name} is as long as one of the two "
                           "clips it joins")
    audio = timeline.get("audio")
    if audio and facts is not None and (facts.get(audio.get("src")) is None):
        out.append(f"the audio file {os.path.basename(str(audio.get('src')))} is missing or unreadable")
    return out


def summary(timeline: dict) -> dict:
    """The timeline as the agent is shown it: one row per clip, and the total."""
    at = starts(timeline)
    rows = []
    for i, c in enumerate(timeline.get("clips") or []):
        rows.append({"n": i + 1, "id": c.get("id"), "file": os.path.basename(str(c.get("src") or "")),
                     "label": c.get("label") or "", "in": c.get("in"), "out": c.get("out"),
                     "length": round(clip_length(c), 3), "starts_at": at[i],
                     "into_it": ("cut" if not float(c.get("transition") or 0) or i == 0
                                 else f"{float(c['transition']):g}s dissolve")})
    audio = timeline.get("audio")
    return {"clips": rows, "duration": round(duration(timeline), 3),
            "fps": timeline.get("fps"), "size": f"{timeline.get('width')}x{timeline.get('height')}",
            "audio": os.path.basename(str(audio["src"])) if audio else "none"}


# ── OpenTimelineIO ───────────────────────────────────────────────────────────
#
# Written as the JSON it is, so no library is needed to hand an edit over. A
# dissolve is laid out as editors expect it: the two clips meet in the middle of
# it, each giving half, with a Transition between them.

def _rt(seconds: float, rate: float) -> dict:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": float(rate),
            "value": float(round(seconds * rate))}


def _range(start: float, length: float, rate: float) -> dict:
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": _rt(start, rate),
            "duration": _rt(length, rate)}


def _url(path: str) -> str:
    p = str(path).replace("\\", "/")
    if p.startswith("//"):
        return "file:" + p
    return "file:///" + p.lstrip("/")


def _otio_clip(name: str, src: str, start: float, length: float, rate: float) -> dict:
    return {"OTIO_SCHEMA": "Clip.1", "name": name, "metadata": {}, "effects": [], "markers": [],
            "source_range": _range(start, length, rate),
            "media_reference": {"OTIO_SCHEMA": "ExternalReference.1", "name": "", "metadata": {},
                                "available_range": None, "target_url": _url(src)}}


def to_otio(timeline: dict, name: str = "agentY edit") -> dict:
    rate = float(timeline.get("fps") or 24.0)
    clips = timeline.get("clips") or []
    video = []
    for i, c in enumerate(clips):
        head = float(c.get("transition") or 0.0) / 2 if i else 0.0
        tail = float(clips[i + 1].get("transition") or 0.0) / 2 if i + 1 < len(clips) else 0.0
        if head:
            video.append({"OTIO_SCHEMA": "Transition.1", "name": "", "metadata": {},
                          "transition_type": "SMPTE_Dissolve",
                          "in_offset": _rt(head, rate), "out_offset": _rt(head, rate)})
        video.append(_otio_clip(c.get("label") or c.get("id") or f"clip {i + 1}", c["src"],
                                float(c.get("in") or 0.0) + head,
                                clip_length(c) - head - tail, rate))

    def track(kind: str, label: str, children: list) -> dict:
        return {"OTIO_SCHEMA": "Track.1", "name": label, "kind": kind, "metadata": {},
                "source_range": None, "effects": [], "markers": [], "children": children}

    tracks = [track("Video", "V1", video)]
    audio = timeline.get("audio")
    if audio:
        tracks.append(track("Audio", "A1", [_otio_clip(
            os.path.basename(str(audio["src"])), audio["src"], float(audio.get("in") or 0.0),
            duration(timeline), rate)]))
    return {"OTIO_SCHEMA": "Timeline.1", "name": name, "metadata": {"made_by": "agentY"},
            "global_start_time": None,
            "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "metadata": {},
                       "source_range": None, "effects": [], "markers": [], "children": tracks}}
