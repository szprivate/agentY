"""Rendering an edit: a timeline (:mod:`src.utils.edit_timeline`) through ffmpeg.

The host does this, not the agent: the agent's part is the timeline, and a
timeline that passes :func:`edit_timeline.problems` renders the same way every
time. Everything is brought to the timeline's size and frame rate first (scaled
to fit, padded, never stretched), then joined - a cut by concatenation, a
dissolve by ``xfade``.

Sound, for now, is a bed laid under the whole film. The clips' own sound is not
carried.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from src.utils import edit_timeline as et

_TIMEOUT = 1800


class RenderError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""


def ffprobe_path() -> str:
    return shutil.which("ffprobe") or ""


# ── what a file is ───────────────────────────────────────────────────────────

def _rate(text: str) -> float:
    try:
        a, _, b = str(text or "").partition("/")
        return round(float(a) / float(b or 1), 3) if float(b or 1) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str) -> dict | None:
    """``{"duration", "fps", "width", "height", "audio", "video", "still"}`` or None.

    None means the file is missing or holds nothing an edit can use.
    """
    path = str(path or "")
    if not path or not os.path.isfile(path):
        return None
    exe = ffprobe_path()
    if not exe:
        raise RenderError("ffprobe is not installed (it comes with ffmpeg) - an edit "
                          "cannot be checked or rendered without it.")
    try:
        run = subprocess.run(
            [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        data = json.loads(run.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video and not audio:
        return None
    try:
        length = float((data.get("format") or {}).get("duration") or (video or audio).get("duration") or 0)
    except (TypeError, ValueError):
        length = 0.0
    still = et.is_still(path)
    return {"duration": 0.0 if still else round(length, 3),
            "fps": 0.0 if still or not video else _rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
            "width": int((video or {}).get("width") or 0), "height": int((video or {}).get("height") or 0),
            "video": bool(video), "audio": bool(audio), "still": still}


def facts_for(timeline: dict, known: dict | None = None) -> dict:
    """``{src: probe}`` for every file the timeline names (None for a bad one)."""
    out = dict(known or {})
    wanted = [c.get("src") for c in timeline.get("clips") or []]
    if timeline.get("audio"):
        wanted.append(timeline["audio"].get("src"))
    for src in wanted:
        if src and src not in out:
            out[src] = probe(src)
    return out


# ── the command ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{float(v):.4f}".rstrip("0").rstrip(".") or "0"


def command(timeline: dict, out_path: str, fmt: str = "mp4", exe: str = "ffmpeg") -> list:
    """The ffmpeg call that renders *timeline* to *out_path*."""
    clips = timeline.get("clips") or []
    if not clips:
        raise RenderError("the timeline has no clips")
    fps = float(timeline.get("fps") or 24.0)
    w, h = int(timeline.get("width") or 1920), int(timeline.get("height") or 1080)
    args = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    parts = []
    for i, c in enumerate(clips):
        length = et.clip_length(c)
        if et.is_still(c["src"]):
            args += ["-loop", "1", "-t", _fmt(length), "-i", c["src"]]
        else:
            args += ["-ss", _fmt(c.get("in") or 0.0), "-t", _fmt(length), "-i", c["src"]]
        parts.append(
            f"[{i}:v]fps={_fmt(fps)},scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
            f"trim=duration={_fmt(length)},setpts=PTS-STARTPTS[v{i}]")
    last, run = "v0", et.clip_length(clips[0])
    for i, c in enumerate(clips[1:], 1):
        d = float(c.get("transition") or 0.0)
        out = f"j{i}"
        if d > 0:
            parts.append(f"[{last}][v{i}]xfade=transition=fade:duration={_fmt(d)}:"
                         f"offset={_fmt(run - d)}[{out}]")
            run += et.clip_length(c) - d
        else:
            parts.append(f"[{last}][v{i}]concat=n=2:v=1:a=0[{out}]")
            run += et.clip_length(c)
        last = out
    audio = timeline.get("audio")
    maps = ["-map", f"[{last}]"]
    if audio:
        n = len(clips)
        args += ["-ss", _fmt(audio.get("in") or 0.0), "-i", audio["src"]]
        fade = min(float(audio.get("fade_out") or 0.0), run)
        chain = f"[{n}:a]atrim=duration={_fmt(run)},asetpts=PTS-STARTPTS,volume={_fmt(audio.get('gain', 1.0))}"
        if fade > 0:
            chain += f",afade=t=out:st={_fmt(run - fade)}:d={_fmt(fade)}"
        parts.append(chain + "[aout]")
        maps += ["-map", "[aout]"]
    args += ["-filter_complex", ";".join(parts)] + maps + ["-r", _fmt(fps), "-t", _fmt(run)]
    if fmt == "mov":
        args += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
        args += ["-c:a", "pcm_s16le"] if audio else ["-an"]
    else:
        args += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart"]
        args += ["-c:a", "aac", "-b:a", "256k"] if audio else ["-an"]
    return args + [out_path]


def render(timeline: dict, out_path: str, fmt: str = "mp4") -> str:
    """Render *timeline*; returns *out_path*. Raises :class:`RenderError` with
    ffmpeg's own words when it fails."""
    exe = ffmpeg_path()
    if not exe:
        raise RenderError("ffmpeg is not installed - an edit cannot be rendered without it.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        run = subprocess.run(command(timeline, out_path, fmt, exe), capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RenderError(f"ffmpeg did not finish within {_TIMEOUT // 60} minutes") from None
    except OSError as exc:
        raise RenderError(f"could not start ffmpeg: {exc}") from None
    if run.returncode != 0 or not os.path.isfile(out_path) or not os.path.getsize(out_path):
        raise RenderError("ffmpeg failed: " + (run.stderr or "").strip()[-800:])
    return out_path


def write_otio(timeline: dict, path: str, name: str = "agentY edit") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(et.to_otio(timeline, name), fh, indent=2)
    return path
