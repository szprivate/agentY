"""Looking at material before cutting it.

A cut made from file names and lengths alone is made blind. Three ways to look,
each a plain measurement of a file (ffmpeg does the reading; nothing here is a
model's opinion):

* :func:`contact_sheet` - frames of a clip at even intervals on one picture, so
  the agent can see what is in it and where, and choose in and out points;
* :func:`shots_of` - where a FILM cuts: its shot lengths, which is the rhythm a
  reference film hands on to a new cut;
* :func:`beats` - where the beats of a piece of music fall, so cuts can land on
  them.
"""

from __future__ import annotations

import math
import os
import re
import subprocess

from src.utils.edit_render import RenderError, ffmpeg_path, probe

_PTS = re.compile(r"pts_time:([0-9.]+)")
_RATE = 22050
_HOP = 512
_FRAME_MIDDLE = _HOP + (_HOP * 4) // 2      # one hop on, plus half an analysis window
_PULSE = 0.3        # how strongly the onsets must repeat to count as a beat


def _run(args: list, timeout: int = 600, binary: bool = False):
    exe = ffmpeg_path()
    if not exe:
        raise RenderError("ffmpeg is not installed - the material cannot be looked at without it.")
    try:
        return subprocess.run([exe, "-hide_banner", *args], capture_output=True, timeout=timeout,
                              **({} if binary else {"text": True, "encoding": "utf-8",
                                                    "errors": "replace"}))
    except subprocess.TimeoutExpired:
        raise RenderError("ffmpeg took too long reading the file") from None
    except OSError as exc:
        raise RenderError(f"could not start ffmpeg: {exc}") from None


# ── seeing a clip ────────────────────────────────────────────────────────────

def sheet_times(length: float, frames: int) -> list:
    """The moments a sheet of *frames* shows: the middle of each equal part."""
    frames = max(1, int(frames))
    return [round(length * (i + 0.5) / frames, 3) for i in range(frames)]


def contact_sheet(path: str, out_path: str, frames: int = 8, columns: int = 4,
                  width: int = 384) -> dict:
    """Write a contact sheet of *path*; returns ``{"sheet", "times", "columns"}``.

    *times* are the seconds the frames were taken at, row by row from the top
    left - what the agent reads an in or out point off.
    """
    fact = probe(path)
    if not fact or not fact.get("video"):
        raise RenderError(f"{os.path.basename(path)} is missing or has no picture")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if fact.get("still") or not fact.get("duration"):
        run = _run(["-y", "-loglevel", "error", "-i", path, "-vf", f"scale={width * 2}:-2",
                    "-frames:v", "1", out_path])
        times, columns = [0.0], 1
    else:
        frames = max(2, min(24, int(frames)))
        columns = max(1, min(frames, int(columns)))
        rows = math.ceil(frames / columns)
        times = sheet_times(float(fact["duration"]), frames)
        # One frame per equal part of the clip, taken from its middle.
        step = float(fact["duration"]) / frames
        vf = (f"fps=1/{step:.6f}:start_time={step / 2:.6f},scale={width}:-2,"
              f"tile={columns}x{rows}:padding=4:margin=4")
        run = _run(["-y", "-loglevel", "error", "-i", path, "-vf", vf, "-frames:v", "1",
                    "-q:v", "3", out_path])
    if run.returncode != 0 or not os.path.isfile(out_path):
        raise RenderError("could not make the contact sheet: " + str(run.stderr or "").strip()[-400:])
    return {"sheet": out_path, "times": times, "columns": columns}


# ── a film's rhythm ──────────────────────────────────────────────────────────

def lengths_from_cuts(cuts: list, total: float, shortest: float = 0.2) -> list:
    """Shot lengths from the moments a film cuts at; cuts closer together than
    *shortest* are one cut seen twice (a flash, a dissolve's two ends)."""
    marks, last = [], 0.0
    for t in sorted(float(c) for c in cuts):
        if 0 < t < total and t - last >= shortest:
            marks.append(t)
            last = t
    edges = [0.0] + marks + [float(total)]
    return [round(b - a, 3) for a, b in zip(edges, edges[1:]) if b - a > 0]


def shots_of(path: str, sensitivity: float = 0.3) -> dict:
    """Where *path* cuts: ``{"shots": [lengths], "cuts": [times], "duration", "average"}``.

    *sensitivity* is ffmpeg's scene score (0-1): lower finds softer cuts and
    more false ones. A dissolve reads as one cut somewhere inside it.
    """
    fact = probe(path)
    if not fact or not fact.get("video") or not fact.get("duration"):
        raise RenderError(f"{os.path.basename(path)} is missing or is not a video")
    run = _run(["-i", path, "-an", "-vf", f"select='gt(scene,{float(sensitivity):.3f})',showinfo",
                "-f", "null", "-"])
    cuts = [float(m.group(1)) for m in _PTS.finditer(run.stderr or "")]
    shots = lengths_from_cuts(cuts, float(fact["duration"]))
    marks, at = [], 0.0
    for s in shots[:-1]:
        at += s
        marks.append(round(at, 3))
    return {"shots": shots, "cuts": marks, "duration": float(fact["duration"]),
            "average": round(sum(shots) / len(shots), 3) if shots else 0.0}


# ── a piece of music's beats ─────────────────────────────────────────────────

def onset_strength(samples, rate: int = _RATE, hop: int = _HOP):
    """How much NEW energy arrives in each hop of *samples* (a numpy array)."""
    import numpy as np
    from scipy.signal import stft
    if len(samples) < hop * 4:
        return np.zeros(0)
    _f, _t, spec = stft(samples.astype("float32"), fs=rate, nperseg=hop * 4, noverlap=hop * 3,
                        boundary=None, padded=False)
    mag = np.log1p(np.abs(spec) * 10.0)
    flux = np.maximum(0.0, mag[:, 1:] - mag[:, :-1]).sum(axis=0)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux


def _peak(corr, at: int) -> float:
    """Where the peak near index *at* really is, between the samples."""
    if at <= 0 or at + 1 >= len(corr):
        return float(at)
    a, b, c = float(corr[at - 1]), float(corr[at]), float(corr[at + 1])
    bend = a - 2 * b + c
    return float(at) if bend >= 0 else at + 0.5 * (a - c) / bend


def tempo_and_beats(strength, rate: int = _RATE, hop: int = _HOP, low: float = 60.0,
                    high: float = 180.0) -> tuple:
    """``(bpm, [beat times])`` from an onset-strength curve; ``(0, [])`` when it
    has no steady pulse to find."""
    import numpy as np
    n = len(strength)
    per_second = rate / hop
    if n < int(per_second * 2):
        return 0.0, []
    x = strength - strength.mean()
    if not np.any(x):
        return 0.0, []
    corr = np.correlate(x, x, mode="full")[n - 1:]
    lo, hi = int(per_second * 60.0 / high), int(per_second * 60.0 / low)
    hi = min(hi, n - 1)
    if hi <= lo:
        return 0.0, []
    lag = lo + int(np.argmax(corr[lo:hi + 1]))
    # A pulse comes back as strongly a period later as it started; noise does
    # not come back at all. Below this there is nothing to follow.
    if corr[lag] < _PULSE * corr[0]:
        return 0.0, []
    # A pulse repeats at every multiple of its period, and the longer one often
    # scores a hair higher: take the faster while it is nearly as strong, or a
    # 120 bpm track is read as 60.
    while lag // 2 >= lo:
        half = max(range(max(lo, lag // 2 - 1), lag // 2 + 2), key=lambda k: float(corr[k]))
        if corr[half] < 0.8 * corr[lag]:
            break
        lag = half
    # The period, finer than one hop: a hop is 23 ms, and a period that is off
    # by half of one drifts a quarter of a second over a twelve-second piece.
    # Measured over several periods where the piece is long enough.
    period = _peak(corr, lag)
    for several in (4, 8, 16):
        far = int(round(period * several))
        if far + 2 >= n // 2:
            break
        best = max(range(far - 2, far + 3), key=lambda k: float(corr[k]))
        if corr[best] < 0.5 * corr[lag]:
            break
        period = _peak(corr, best) / several
    bpm = 60.0 * per_second / period
    # Where the grid sits: the offset whose beats land on the most onset energy.
    def energy(offset: float) -> float:
        idx = np.round(np.arange(offset, n - 1, period)).astype(int)
        return float(strength[idx].sum())
    offset = max((o * 0.5 for o in range(int(period * 2))), key=energy)
    # Entry i of the curve is the change INTO frame i+1, and a frame is dated by
    # its middle: without this every beat is called about 0.07 s early, which is
    # two frames of picture.
    late = _FRAME_MIDDLE / rate
    times = [round(float(v) / per_second + late, 3) for v in np.arange(offset, n - 1, period)]
    return round(bpm, 1), times


def beats(path: str, limit: int = 400) -> dict:
    """``{"bpm", "beats": [seconds], "duration"}`` for a sound or video file."""
    import numpy as np
    fact = probe(path)
    if not fact or not fact.get("audio"):
        raise RenderError(f"{os.path.basename(path)} is missing or has no sound")
    run = _run(["-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(_RATE),
                "-f", "s16le", "-"], binary=True)
    if run.returncode != 0 or not run.stdout:
        raise RenderError("could not read the sound of " + os.path.basename(path))
    samples = np.frombuffer(run.stdout, dtype="<i2").astype("float32") / 32768.0
    bpm, times = tempo_and_beats(onset_strength(samples))
    return {"bpm": bpm, "beats": times[:max(1, int(limit))],
            "duration": round(len(samples) / _RATE, 3),
            **({} if bpm else {"note": "no steady pulse was found - there are no beats to cut on"})}
