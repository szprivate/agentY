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
     "overlays": [{"id": "t1", "preset": "title_rise", "text": "…", "start": 0.5,
                   "length": 3.0, "params": {"size": 96}}, ...]}

An overlay is TEXT laid over the picture: a preset (:mod:`src.utils.text_presets`)
says how it looks and moves, the overlay says what it reads, when, and for how
long. ``start`` is a moment in the FILM, not in a clip.

A clip's own sound comes with it (``clip_sound``, on unless switched off), at the
clip's ``volume`` (1 = as it is, 0 = silent); a clip with no sound is silence
for its length. ``audio`` is a bed laid under the whole film, mixed with it.

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
            "clips": [], "audio": None, "overlays": [], "clip_sound": True}


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
    elif kind == "sound":
        _sound(tl, op)
    elif kind == "text":
        _text(tl, op)
    elif kind == "text_remove":
        tl["overlays"].pop(_find_text(tl, op.get("text_id") or op.get("id")))
    elif kind == "rhythm":
        rhythm(tl, op.get("lengths"), facts)
    elif kind == "snap":
        snap(tl, op.get("times"), facts,
             _seconds(op.get("within") if op.get("within") is not None else 0.5, "within"))
    elif kind == "settings":
        for key, cast in (("fps", float), ("width", int), ("height", int)):
            if op.get(key):
                tl[key] = cast(op[key])
    elif kind == "clear":
        tl["clips"], tl["audio"], tl["overlays"] = [], None, []
    else:
        raise EditError("unknown operation - use add, trim, move, replace, remove, "
                        "transition, label, audio, sound, fit, rhythm, snap, text, "
                        "text_remove, settings or clear")


def _sound(tl: dict, op: dict) -> None:
    """The clips' own sound: on or off for the film, or one clip's volume."""
    if op.get("clip") is None:
        if op.get("on") is None and op.get("volume") is None:
            raise EditError("say on: true/false for the whole film, or name a clip and its volume")
        if op.get("on") is not None:
            tl["clip_sound"] = bool(op["on"]) if not isinstance(op["on"], str) \
                else op["on"].strip().lower() in ("true", "on", "yes", "1")
        if op.get("volume") is not None:
            for c in tl["clips"]:
                c["volume"] = _volume(op["volume"])
        return
    c = tl["clips"][_find(tl, op.get("clip"))]
    if op.get("volume") is None:
        raise EditError("give the clip's volume (1 = as it is, 0 = silent)")
    c["volume"] = _volume(op["volume"])


def _volume(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise EditError(f"volume must be a number, not {value!r}") from None
    if not 0 <= v <= 4:
        raise EditError(f"volume {v:g} is out of range (0 = silent, 1 = as it is, at most 4)")
    return round(v, 3)


def _find_text(tl: dict, text_id) -> int:
    want = str(text_id or "").strip()
    for i, o in enumerate(tl.get("overlays") or []):
        if str(o.get("id")) == want:
            return i
    raise EditError(f"there is no text '{text_id}' - the texts are "
                    + (", ".join(str(o.get("id")) for o in tl.get("overlays") or []) or "none yet"))


def _text(tl: dict, op: dict) -> None:
    """Add a text, or - with ``text_id`` - change the one that is there."""
    texts = tl.setdefault("overlays", [])
    if op.get("text_id") or op.get("id"):
        o = texts[_find_text(tl, op.get("text_id") or op.get("id"))]
    else:
        if not str(op.get("text") or "").strip():
            raise EditError("give the words (text)")
        if not str(op.get("preset") or "").strip():
            raise EditError("name the preset (list_text_presets shows them)")
        used = {str(t.get("id")) for t in texts}
        n = len(used) + 1
        while f"t{n}" in used:
            n += 1
        o = {"id": f"t{n}", "preset": "", "text": "", "start": 0.0, "length": 3.0, "params": {}}
        texts.append(o)
    if op.get("text") is not None and str(op.get("text")).strip():
        o["text"] = str(op["text"])
    if op.get("preset") is not None and str(op.get("preset")).strip():
        o["preset"] = str(op["preset"]).strip()
    if op.get("start") is not None:
        o["start"] = round(_seconds(op["start"], "start"), 4)
    if op.get("length") is not None:
        length = _seconds(op["length"], "length")
        if length < 0.2:
            raise EditError(f"a text {length:g}s long cannot be read")
        o["length"] = round(length, 4)
    if isinstance(op.get("params"), dict):
        o["params"] = dict(o.get("params") or {}, **op["params"])


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


def _room(clip: dict, facts: dict | None) -> float:
    """How long *clip* could be from its in point: to the end of its file, when known."""
    if is_still(clip.get("src")):
        return float("inf")
    known = float(((facts or {}).get(clip.get("src")) or {}).get("duration") or 0.0)
    return max(_MIN_CLIP, known - float(clip.get("in") or 0.0)) if known else float("inf")


def rhythm(timeline: dict, lengths, facts: dict | None = None) -> dict:
    """Give the clips, in order, the shot lengths of another film.

    *lengths* is that film's shots in seconds (``edit_look.shots_of``). The
    pattern repeats when the cut has more clips than the reference had shots. A
    clip too short for its slot keeps all it has: the pattern is a guide, and a
    clip is never stretched.
    """
    try:
        pattern = [float(v) for v in (lengths or []) if float(v) >= _MIN_CLIP]
    except (TypeError, ValueError):
        raise EditError("lengths must be a list of seconds") from None
    if not pattern:
        raise EditError("give the reference's shot lengths (lengths: [seconds, …])")
    for i, c in enumerate(timeline.get("clips") or []):
        want = min(pattern[i % len(pattern)], _room(c, facts))
        c["out"] = round(float(c.get("in") or 0.0) + max(_MIN_CLIP, want), 4)
    return timeline


def snap(timeline: dict, times, facts: dict | None = None, within: float = 0.5) -> dict:
    """Move each cut to the nearest of *times* (beats), when one is *within* reach.

    Done from the first cut to the last, by making the clip before the cut longer
    or shorter, so every later cut is measured from where the earlier ones ended
    up. A cut with no beat near it stays where it is.
    """
    try:
        marks = sorted(float(v) for v in (times or []))
    except (TypeError, ValueError):
        raise EditError("times must be a list of seconds") from None
    if not marks:
        raise EditError("give the moments to cut on (times: [seconds, …])")
    clips = timeline.get("clips") or []
    at = 0.0
    for i, c in enumerate(clips):
        if i:
            at -= float(c.get("transition") or 0.0)
        end = at + clip_length(c)
        if i + 1 < len(clips):
            near = min(marks, key=lambda m: abs(m - end))
            length = near - at
            if abs(near - end) <= within and _MIN_CLIP <= length <= _room(c, facts):
                c["out"] = round(float(c.get("in") or 0.0) + length, 4)
                end = near
        at = end
    return timeline


def problems(timeline: dict, facts: dict | None = None, presets=None) -> list:
    """What stops this timeline from being rendered, in words (empty = nothing).

    *presets* - the names of the text presets there are; with it a text naming
    one that does not exist is caught here.
    """
    out = []
    clips = timeline.get("clips") or []
    if not clips:
        return ["the timeline has no clips yet"]
    film = duration(timeline)
    for o in timeline.get("overlays") or []:
        name = f"text {o.get('id')} (\"{str(o.get('text') or '')[:30]}\")"
        if presets is not None and str(o.get("preset")) not in set(presets):
            out.append(f"{name} names the preset '{o.get('preset')}', which does not exist")
        if float(o.get("start") or 0) >= film:
            out.append(f"{name} starts at {float(o.get('start') or 0):g}s, after the film has "
                       f"ended ({film:g}s)")
        elif float(o.get("start") or 0) + float(o.get("length") or 0) > film + 0.05:
            out.append(f"{name} runs to {float(o.get('start') or 0) + float(o.get('length') or 0):g}s "
                       f"but the film ends at {film:g}s")
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
                                 else f"{float(c['transition']):g}s dissolve"),
                     **({"volume": c["volume"]} if c.get("volume") not in (None, 1, 1.0) else {})})
    audio = timeline.get("audio")
    texts = [{"id": o.get("id"), "preset": o.get("preset"), "text": o.get("text"),
              "from": o.get("start"), "to": round(float(o.get("start") or 0) + float(o.get("length") or 0), 3),
              **({"params": o["params"]} if o.get("params") else {})}
             for o in timeline.get("overlays") or []]
    return {"clips": rows, "duration": round(duration(timeline), 3),
            "fps": timeline.get("fps"), "size": f"{timeline.get('width')}x{timeline.get('height')}",
            "audio": os.path.basename(str(audio["src"])) if audio else "none",
            "clip_sound": "on" if timeline.get("clip_sound", True) else "off",
            **({"texts": texts} if texts else {})}


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
    # A text is over the film at a moment, not inside a clip: a marker on the
    # whole timeline, where an editor sees it as a note of what goes there and
    # the viewer draws it as a track of its own.
    texts = [{"OTIO_SCHEMA": "Marker.2", "name": str(o.get("text") or "").splitlines()[0][:80]
              if str(o.get("text") or "").strip() else "", "color": "PURPLE",
              "marked_range": _range(float(o.get("start") or 0.0), float(o.get("length") or 0.0), rate),
              "metadata": {"agentY": {"kind": "text", "text": str(o.get("text") or ""),
                                      "preset": str(o.get("preset") or ""),
                                      "params": dict(o.get("params") or {})}}}
             for o in timeline.get("overlays") or []]
    return {"OTIO_SCHEMA": "Timeline.1", "name": name, "metadata": {"made_by": "agentY"},
            "global_start_time": None,
            "tracks": {"OTIO_SCHEMA": "Stack.1", "name": "tracks", "metadata": {},
                       "source_range": None, "effects": [], "markers": texts, "children": tracks}}
