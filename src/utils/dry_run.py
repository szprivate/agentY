"""Do all the thinking, build every graph, and submit none of them.

A hook workflow is two things at once: a chain of reasoning (each hook reads its
inputs, writes values, and hands them to the next) and a pile of paid API calls.
Only the second one is expensive, and only the first one is usually wrong. Before
this, the only way to find out whether a five-hook chain wired up the way you
meant was to run it and watch the invoice.

A dry run keeps everything up to the submission. The hooks are read, the
directives are answered, the values are written, the variants are built and
written to disk as real workflow files you can open — and then, where ComfyUI
would be handed the graph, each variant is answered with a **stand-in**: a file
path, and nothing else. No file is created. Nothing is staged onto the canvas,
because there is nothing to stage.

The stand-ins matter more than they look. A later hook whose directive is *"take
the reference frames you just made and queue one video per shot"* has to receive
*something* where the references were, or the half of the chain that is most
worth testing never runs. It gets paths — named after the variant that would have
produced them, so "reference 3" still reads as "Ben, grey suit" — and the tools
that would otherwise open them (analysis, upload) recognise a stand-in and answer
in kind rather than failing on a file that was never written.

State is per turn and process-wide, like ``workflow_signal``: the tools that need
to consult it are module-level, not closures over the pipeline.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading

# In the path, in the filename, in every report: a stand-in must never be
# mistaken for a render, by a person or by an agent reading its own transcript.
MARKER = "DRY-RUN"
_DIRNAME = "agenty_dry_run"

_lock = threading.Lock()
_on: bool = False
_runs: list[dict] = []          # one per workflow that WOULD have been submitted
_stand_ins: dict[str, str] = {}  # stand-in path -> what it stands for


def arm(on: bool = True) -> None:
    """Turn the current turn into a dry run (or back into a real one)."""
    global _on
    with _lock:
        _on = bool(on)
        if _on:
            _runs.clear()
            _stand_ins.clear()


def active() -> bool:
    """Whether this turn is a dry run."""
    with _lock:
        return _on


def reset() -> None:
    """Disarm and forget — called at the end of every turn, dry or not."""
    global _on
    with _lock:
        _on = False
        _runs.clear()
        _stand_ins.clear()


# ── what a graph would have produced ─────────────────────────────────────────

# Judged by class name alone, deliberately: this runs before anything else and
# must not depend on a ComfyUI being up. It only decides an extension.
_VIDEO_HINTS = ("video", "vhs_", "animatediff", "svd", "kling", "veo", "seedance",
                "wanvideo", "hunyuanvideo", "ltxv", "mochi", "cogvideo", "minimax")
_AUDIO_HINTS = ("audio", "vocal", "music", "tts", "speech", "elevenlabs")
_SAVE_HINTS = ("save", "videocombine", "sendto", "output")

_EXT = {"video": "mp4", "audio": "wav", "image": "png"}


def _classes(prompt: dict | None) -> list[str]:
    return [str(n.get("class_type") or "").lower()
            for n in (prompt or {}).values() if isinstance(n, dict)]


def media_kind(prompt: dict | None) -> str:
    """``"image"`` | ``"video"`` | ``"audio"`` — what this graph makes.

    Video wins over audio wins over image, because a graph that makes a video
    usually also holds the audio and image nodes that went into it.
    """
    classes = _classes(prompt)
    if any(h in c for c in classes for h in _VIDEO_HINTS):
        return "video"
    if any(h in c for c in classes for h in _AUDIO_HINTS):
        return "audio"
    return "image"


def _file_count(prompt: dict | None) -> int:
    """Roughly how many files this graph writes: one per save-ish node.

    An approximation, and named as one. Getting it exactly right needs the run
    itself (batch sizes, a video node's frame policy); getting it roughly right
    is enough for the thing this is for — letting the next hook see that there
    were three references rather than one.
    """
    n = sum(1 for c in _classes(prompt) if any(h in c for h in _SAVE_HINTS))
    return max(1, min(n, 4))


def slug(text: str, limit: int = 40) -> str:
    """A filename-safe stub of *text* — used for stand-in names and sidebar entries."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s[:limit].strip("-")


def stand_ins(prompt: dict | None, workflow_path: str, *, label: str = "",
              index: int = 1) -> list[str]:
    """The path(s) *workflow_path* would have produced, had it been submitted.

    Named after the variant, not the file that never existed: a batch of five
    references is five different things, and a stand-in called ``DRY-RUN_003.png``
    would throw away the one fact — which is which — that the rest of the chain
    depends on.
    """
    ext = _EXT.get(media_kind(prompt), "png")
    base = os.path.join(tempfile.gettempdir(), _DIRNAME)
    name = slug(label)
    count = _file_count(prompt)
    out: list[str] = []
    for i in range(count):
        part = f"_{i + 1}" if count > 1 else ""
        fname = f"{MARKER}_{index:03d}{part}" + (f"_{name}" if name else "") + f".{ext}"
        path = os.path.join(base, fname)
        out.append(path)
        with _lock:
            _stand_ins[_key(path)] = " ".join(str(label or "").split())[:120]
    return out


def _key(path) -> str:
    return str(path or "").replace("/", os.sep).lower()


def is_stand_in(path) -> bool:
    """Whether *path* is one of this turn's stand-ins.

    The registry answers for anything produced here. The marker in the filename
    is the backstop for a path that came back through an agent's own text (they
    are quoted, re-typed and pasted between tools), where the exact spelling of a
    Windows path is not something to bet on.
    """
    p = str(path or "")
    if not p:
        return False
    with _lock:
        if _key(p) in _stand_ins:
            return True
    return MARKER in os.path.basename(p)


def stands_for(path) -> str:
    """What the stand-in was going to be — the variant's own label, if it had one."""
    with _lock:
        return _stand_ins.get(_key(path), "")


def record(workflow_path: str, outputs: list, *, label: str = "",
           kind: str = "", what: str = "") -> dict:
    """Note that *workflow_path* was built and deliberately not submitted."""
    entry = {"workflow": str(workflow_path), "outputs": [str(o) for o in (outputs or [])],
             "label": " ".join(str(label or "").split())[:120],
             "kind": kind or "", "what": what or ""}
    with _lock:
        _runs.append(entry)
    return entry


def runs() -> list:
    """Every graph this turn built and did not submit, in the order it built them."""
    with _lock:
        return [dict(r) for r in _runs]


def summary() -> str:
    """The end-of-turn block, or '' when a dry run built nothing at all."""
    made = runs()
    if not made:
        return ""
    n_out = sum(len(r["outputs"]) for r in made)
    lines = [f"🧪 DRY RUN — {len(made)} workflow(s) built, nothing submitted to ComfyUI. "
             f"{n_out} stand-in output(s); no image or video was generated."]
    for i, r in enumerate(made, 1):
        what = r["label"] or r["what"] or os.path.basename(r["workflow"])
        lines.append(f"   {i}. {what}")
        lines.append(f"      graph: {r['workflow']}")
        for o in r["outputs"]:
            lines.append(f"      would produce: {os.path.basename(o)}")
    lines.append("   Open any of those graph files to check the wiring and the values "
                 "that were written into it. Re-run with Full run to generate for real.")
    return "\n".join(lines)


def stand_in_notice(path) -> str:
    """What a tool should say when handed a stand-in instead of a real file."""
    what = stands_for(path)
    return ("DRY RUN — this path is a stand-in for a generation that was deliberately "
            "not run, so there is no file to read"
            + (f'. It stands for: "{what}"' if what else "")
            + ". Treat the generation as having succeeded and carry on with the next "
              "step; do not try to open, re-generate or repair it.")
