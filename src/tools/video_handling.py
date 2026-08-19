"""agentY video-understanding tool.

``analyze_video`` samples frames from a video and hands them, in order, to a
stateless Video Agent (default Qwen2.5-VL on Alibaba Model Studio / DashScope).
Qwen-VL reads a sequence of images as video frames, so we reuse the same Strands
image content blocks the vision path uses — no special "video" content block is
needed. Frame extraction uses OpenCV (``cv2``, already present via ComfyUI).

Mirrors :mod:`src.tools.image_handling`'s ``analyze_image`` / ``set_vision_agent``
pattern: the pipeline registers a shared Video Agent at init, and the tool clears
history on every call so each analysis is independent.
"""
from __future__ import annotations

import io as _io
import os
from pathlib import Path
from typing import Optional

from strands import Agent, tool

from src.tools.image_handling import _MAX_IMAGE_BYTES, AgentPool, _downsize
# Shot detection/splitting is shared with agentY-mcp, so it lives in the core
# layer; only the dry-run behaviour below is agentY's.
from agenty_core.tools.video import (
    _MIN_SHOT_SECONDS,
    split_video_into_shots as _core_split_video_into_shots,
)

_video_agent: Optional[Agent] = None
_video_pool: Optional[AgentPool] = None


def set_video_agent(agent: Agent, factory=None, max_parallel: int = 1) -> None:
    """Register the Video Agent used by :func:`analyze_video`.

    Same shape as ``set_vision_agent``: one Strands agent cannot serve two calls
    at once, so concurrent ``analyze_video`` calls need one instance each. See
    :class:`~src.tools.image_handling.AgentPool`. Each call here carries several
    sampled frames, so the default cap is deliberately lower than vision's.
    """
    global _video_agent, _video_pool
    _video_agent = agent
    _video_pool = AgentPool(agent, factory=factory, size=max_parallel)


def _ensure_video_pool() -> Optional[AgentPool]:
    """The pool wrapping the registered video agent, built on demand."""
    global _video_pool
    if _video_agent is None:
        return None
    if _video_pool is None or _video_pool.primary is not _video_agent:
        _video_pool = AgentPool(_video_agent)
    return _video_pool


def video_agents() -> list[Agent]:
    """Every live video agent, so the pipeline can price all of their tokens."""
    pool = _ensure_video_pool()
    return pool.instances() if pool else []


def _resolve_local_video(path: str) -> Optional[str]:
    """Resolve a video path: direct/absolute, ``~``-expanded, then ComfyUI's input
    dir by basename (a bare loader filename). Returns the absolute path or None."""
    raw = (path or "").strip().strip('"')
    if not raw:
        return None
    p = Path(os.path.expanduser(raw))
    if p.is_file():
        return str(p.resolve())
    try:
        from src.tools.image_handling import _comfy_input_dir
        d = _comfy_input_dir()
        if d:
            cand = Path(d) / p.name
            if cand.is_file():
                return str(cand.resolve())
    except Exception:  # noqa: BLE001
        pass
    return None


def _extract_frames(path: str, max_frames: int) -> tuple[list[bytes], dict]:
    """Sample up to *max_frames* frames evenly across the clip.

    Returns ``(jpeg_bytes_list, meta)`` where meta carries total_frames / fps /
    duration_s. Each frame is JPEG-encoded and passed through ``_downsize`` so it
    satisfies the same size/px limits as image inputs.
    """
    import cv2
    from PIL import Image as _PILImage

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open video (unsupported codec or unreadable file)")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total / fps) if (total and fps) else 0.0

    if total > 0:
        if max_frames <= 1:
            idxs = [0]
        else:
            idxs = [round(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
        idxs = sorted({int(i) for i in idxs})
    else:
        idxs = list(range(max_frames))  # unknown length (stream): read sequentially

    frames: list[bytes] = []
    for idx in idxs:
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = _io.BytesIO()
        _PILImage.fromarray(rgb).save(buf, format="JPEG", quality=85)
        try:
            data, _fmt = _downsize(buf.getvalue(), "jpeg")
        except Exception:  # noqa: BLE001
            data = buf.getvalue()
        if len(data) <= _MAX_IMAGE_BYTES:
            frames.append(data)
    cap.release()
    return frames, {"total_frames": total, "fps": round(fps, 2), "duration_s": round(duration, 2)}


@tool
def analyze_video(file_path: str = "", question: str = "", max_frames: int = 8) -> dict:
    """Analyse a video by sampling frames and describing them with the Video Agent.

    Frames are sampled evenly across the whole clip and sent, in order, to a
    vision-language model (default Qwen2.5-VL) that reads them as a video. Use this
    to understand the CONTENT of a video input (e.g. a clip wired from an agentY
    video collector) — subject, action, motion, scene, style — before choosing or
    building a workflow. No frame bytes enter your own context window.

    Args:
        file_path: Absolute or relative path to a local video file
            (mp4/mov/webm/mkv/avi/m4v/mpg).
        question:  What to focus on (e.g. "what happens in this clip?",
            "describe the camera motion and style"). Defaults to a general summary.
        max_frames: How many frames to sample (1–32; default 8). More frames give
            finer temporal detail at higher token cost. Capped by
            ``AGENTY_VIDEO_MAX_FRAMES`` when set.

    Returns:
        ``{"status": "success"|"error", "content": [{"text": ...}]}``.
    """
    # A dry run's "generations" are paths and nothing else — see analyze_image.
    try:
        from src.utils import dry_run as _dry
        if _dry.active() and _dry.is_stand_in(file_path):
            return {"status": "ok", "content": [{"text": _dry.stand_in_notice(file_path)}]}
    except Exception:  # noqa: BLE001
        pass
    # Test/speed mode: skip the model and return a canned description.
    if os.environ.get("AGENTY_STUB_VISION") or os.environ.get("AGENTY_STUB_VIDEO"):
        label = os.path.basename(file_path) or "input video"
        return {"status": "ok", "content": [{"text": (
            f"Video analysis for {label} (stubbed): a short clip with a single moving "
            "subject on a simple background, steady camera, natural lighting."
        )}]}

    resolved = _resolve_local_video(file_path)
    if resolved is None:
        return {"status": "error", "content": [{"text": f"Video not found: {file_path}"}]}

    try:
        n = max(1, min(int(max_frames or 8), 32))
    except (TypeError, ValueError):
        n = 8
    env_cap = os.environ.get("AGENTY_VIDEO_MAX_FRAMES")
    if env_cap:
        try:
            n = max(1, min(n, int(env_cap)))
        except ValueError:
            pass

    try:
        frames, meta = _extract_frames(resolved, n)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"Could not read video {resolved}: {exc}"}]}
    if not frames:
        return {"status": "error", "content": [{"text": (
            f"No frames could be sampled from {resolved} (empty or unsupported video)."
        )}]}

    if _video_agent is None:
        return {"status": "error", "content": [{"text": (
            "No Video Agent is registered — call set_video_agent() during pipeline init."
        )}]}

    dur = float(meta.get("duration_s") or 0.0)
    header = (
        f"These {len(frames)} frames were sampled evenly, in order, from a video "
        f"(~{dur:.1f}s, {meta.get('fps', 0)} fps). Read them as consecutive video "
        f"frames. {question or 'Describe what happens in the video in detail.'}"
    )
    user_message: list = [{"image": {"format": "jpeg", "source": {"bytes": f}}} for f in frames]
    user_message.append({"text": header})

    try:
        with _ensure_video_pool().borrow() as _agent:
            _agent.messages.clear()
            result = str(_agent(user_message))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"Video analysis failed: {exc}"}]}

    return {"status": "success", "content": [{"text": (
        f"Video analysis for {resolved} ({len(frames)} frames, ~{dur:.1f}s):\n\n{result}"
    )}]}


@tool
def split_video_into_shots(file_path: str = "", detector: str = "content",
                           threshold: float = 27.0,
                           min_shot_seconds: float = _MIN_SHOT_SECONDS,
                           detect_only: bool = False, output_dir: str = "",
                           fast: bool = False, max_shots: int = 0) -> dict:
    """Detect the cuts in a video and split it into one file per shot.

    Finds shot boundaries automatically and writes each shot as its own video
    file, so a clip can be worked on shot by shot — restyle one shot, feed each to
    a per-shot workflow, or just find out where the cuts are. Run it with
    ``detect_only=True`` first when unsure of the settings: that reads the file
    and writes nothing.

    Args:
        file_path: The video to split. A path, or a bare filename sitting in
            ComfyUI's input dir.
        detector: ``content`` (default) compares consecutive frames — right for
            edited footage with hard cuts. ``adaptive`` scores against a rolling
            window instead, which stops fast camera motion, whip pans and strobes
            from reading as cuts; use it on handheld or high-motion material.
        threshold: Sensitivity. LOWER finds more cuts. 27.0 suits ``content``; for
            ``adaptive`` the comparable default is 3.0. If a known cut is missed,
            lower it; if one shot comes back split into several, raise it.
        min_shot_seconds: Ignore any shot shorter than this (default 0.4), so a
            flash frame or a one-frame glitch is not reported as a shot.
        detect_only: Report where the cuts are and write nothing.
        output_dir: Where the shots go. Defaults to a ``<name>_shots`` folder
            under the agent's videos directory.
        fast: Stream-copy instead of re-encoding. LEAVE THIS OFF unless you know
            the source is keyframe-dense: a copy can only start on a keyframe, and
            generated video (ComfyUI's savers, most model APIs) is usually written
            with a single keyframe at the start, which makes every shot run from
            the beginning of the file. That is detected and re-cut properly rather
            than returned, so the cost of asking for it wrongly is wasted time,
            not wrong files — but it buys nothing on that footage.
        max_shots: Stop writing after this many (default 200). Detection still
            reports everything it found.

    Returns:
        ``{"status", "content": [{"text"}], "shots": [...], "output_dir", "meta"}``
        — each shot carrying its index, start/end in seconds and timecode, its
        duration, and (unless ``detect_only``) the path written.
    """
    # The whole tool is agenty_core's, so agentY-mcp can offer the same one. What
    # is agentY's and stays here: a dry run's "generations" are paths with no file
    # behind them, and a chain like "make the video, then split it into shots"
    # must not report itself broken because the second stage could not open the
    # first stage's stand-in. Same contract analyze_video/analyze_image keep.
    try:
        from src.utils import dry_run as _dry
        if _dry.active() and _dry.is_stand_in(file_path):
            return {"status": "ok", "shots": [], "meta": {}, "output_dir": "",
                    "content": [{"text": _dry.stand_in_notice(file_path)}]}
    except Exception:  # noqa: BLE001
        pass
    return _core_split_video_into_shots(
        file_path=file_path, detector=detector, threshold=threshold,
        min_shot_seconds=min_shot_seconds, detect_only=detect_only,
        output_dir=output_dir, fast=fast, max_shots=max_shots)
