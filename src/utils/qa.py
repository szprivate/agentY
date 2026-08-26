"""Output QA: the briefing the user writes, and the check that judges against it.

agentY can generate a thing; it could not, until now, tell you whether the thing
is any *good* by your standards. This module owns the user-facing half of that —
the **QA briefing** — plus the call that judges one finished file against it.

A briefing is deliberately two things at once, because "is this right?" usually
is: **criteria** (prose or bullets — "skin tones warm not orange", "no visible
extra fingers") and **references** (mood images the output should sit next to
without looking out of place). Text alone can't express a grade; a mood board
alone can't express a rule.

Three ways to write one, all producing the same object:

* **A canvas hook** with ``purpose: "qa"`` — the directive is the checklist and
  the hook's *anchors* are the references. This is the primary surface, because
  wiring an image into an anchor is what makes it unambiguously a REFERENCE and
  not another input to the workflow. That distinction is the whole problem: a
  turn can carry inputs, outputs and references at once, and prose can't reliably
  keep them apart.
* **A named file** — ``<briefing_dir>/<name>.md`` for the criteria plus an
  optional sibling ``<name>.refs/`` folder of mood images. Reusable across graphs
  and threads, and it lives in version control.
* **``/qa`` in the chat panel** — a briefing attached to the conversation, for
  when there is no canvas graph in play.

They compose: a hook or a ``/qa`` briefing may cite ``@name`` to pull a named
file in on top of its own text.

Nothing here runs unless a briefing exists. The ``[qa]`` settings decide *how* QA
runs (retries, caps, which model), never *whether*.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff",
              ".mp4", ".mov", ".webm", ".mkv", ".avi"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}

# agentY collector nodes hold their file list in a `files` widget, one path per line.
_COLLECTOR_TYPES = {"AgentYImageCollector", "AgentYVideoCollector"}

# A briefing cites another by name with @name — resolved against briefing_dir.
_CITE_RE = re.compile(r"(?:^|\s)@([A-Za-z0-9][A-Za-z0-9._-]*)")


# ── settings ────────────────────────────────────────────────────────────────────

def qa_settings() -> dict:
    """The effective ``[qa]`` settings, with sane values for missing/invalid keys."""
    try:
        from src.utils.settings import load_settings
        raw = load_settings().get("qa") or {}
    except Exception:  # noqa: BLE001 — never let a bad settings file break a turn
        raw = {}

    def _int(key: str, default: int, low: int = 0) -> int:
        try:
            return max(low, int(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    enabled = bool(raw.get("enabled", True))
    env = os.environ.get("AGENTY_QA")
    if env is not None and env.strip() != "":
        enabled = env.strip().lower() not in ("0", "false", "no", "off")
    return {
        "enabled": enabled,
        "max_retries": _int("max_retries", 1),
        "max_outputs": _int("max_outputs", 6, low=1),
        "max_references": _int("max_references", 4),
        "video_frames": _int("video_frames", 3, low=1),
        "briefing_dir": str(raw.get("briefing_dir") or "./config/qa/"),
    }


def briefing_dir() -> Path:
    """Absolute path of the named-briefing folder (may not exist yet)."""
    raw = qa_settings()["briefing_dir"]
    p = Path(raw)
    return p if p.is_absolute() else (_PROJECT_ROOT / raw).resolve()


# ── the briefing ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QaBriefing:
    """What the QA agent judges against: criteria plus reference images.

    ``sources`` records where it came from (hook / file:<name> / thread), purely so
    the chat panel and logs can say *why* QA is running — a briefing that silently
    turns on is a briefing the user will curse at.
    """
    criteria: str = ""
    reference_paths: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    # What a failing verdict should cause, written by the user in their own
    # briefing: how many times to try again, and — when the fix is not "roll the
    # same generation again" but "go back a stage" — which hook to re-enter.
    retry_budget: int | None = None
    retry_hook: str = ""

    def __bool__(self) -> bool:
        return bool(self.criteria.strip() or self.reference_paths)

    def merged_with(self, other: "QaBriefing | None") -> "QaBriefing":
        """This briefing with *other* folded in (other's criteria appended)."""
        if not other:
            return self
        criteria = "\n".join(t for t in (self.criteria.strip(), other.criteria.strip()) if t)
        refs = list(self.reference_paths)
        refs += [p for p in other.reference_paths if p not in set(refs)]
        return QaBriefing(criteria=criteria, reference_paths=tuple(refs),
                          sources=tuple(dict.fromkeys(self.sources + other.sources)),
                          retry_budget=(self.retry_budget if self.retry_budget is not None
                                        else other.retry_budget),
                          retry_hook=self.retry_hook or other.retry_hook)

    def describe(self) -> str:
        """One line for the chat panel: what is being enforced and from where."""
        bullets = len([ln for ln in self.criteria.splitlines() if ln.strip()])
        where = ", ".join(self.sources) or "unknown"
        refs = len(self.reference_paths)
        ref_txt = f", {refs} reference image{'' if refs == 1 else 's'}" if refs else ""
        retry = ""
        if self.retry_hook:
            retry = f", re-run hook {self.retry_hook} on a fail"
        elif self.retry_budget is not None:
            retry = f", {self.retry_budget} retr{'y' if self.retry_budget == 1 else 'ies'}"
        return (f"{bullets} criteri{'on' if bullets == 1 else 'a'}{ref_txt}{retry} "
                f"(from {where})")


# What the user writes in a qa briefing to say what a failure should cause.
# Deliberately a small, stated syntax rather than an inference: "retry" appearing
# in prose ("no retry-looking artefacts") must not silently change the budget.
_RETRY_PATTERNS = (
    re.compile(r"\bre-?try\s*[:=]\s*hook\s*#?(?P<hook>\d+)(?:\s*[x×]\s*(?P<n>\d+))?", re.I),
    re.compile(r"\bre-?run\s*[:=]?\s*hook\s*#?(?P<hook>\d+)(?:\s*[x×]\s*(?P<n>\d+))?", re.I),
    re.compile(r"\bre-?try\s*[:=]\s*(?P<n>\d+)", re.I),
)


def parse_retry(text: str) -> tuple[int | None, str]:
    """``(budget, hook_id)`` from a briefing's own words, or ``(None, "")``.

    ``retry: 2`` bounds the automatic re-generation of the same graph. ``retry:
    hook 5`` says the fix lives a stage earlier — regenerate the reference, not
    the shot that used it — which the runtime cannot do by itself (that stage is
    an agent writing prompts), so it is handed to the agent as an instruction
    with the failed outputs attached.
    """
    body = str(text or "")
    for pat in _RETRY_PATTERNS:
        m = pat.search(body)
        if m:
            n = m.groupdict().get("n")
            hook = m.groupdict().get("hook") or ""
            return (int(n) if n else None), str(hook)
    return None, ""


# ── surface 1: the `qa` canvas hook ─────────────────────────────────────────────

def anchor_media_paths(anchor: dict, resolver=None) -> list[str]:
    """Every media file a hook anchor points at, as absolute paths.

    Covers the three ways an anchor can carry media, so a QA hook's references can
    be wired however is convenient: an agentY **collector** (its explicit file
    list), a **tapped tensor** (a mid-graph wire :mod:`src.utils.canvas_tap`
    already rendered to disk this turn), or an ordinary **loader** whose widget
    names a file. *resolver* is the caller's ``value, kind -> abs path | None``.
    """
    out: list[str] = []

    def _add(path: str) -> None:
        p = (path or "").strip().strip('"')
        if p and p not in out:
            out.append(p)

    if str(anchor.get("type") or "") in _COLLECTOR_TYPES:
        files = (anchor.get("widgets") or {}).get("files")
        for line in str(files or "").splitlines():
            resolved = resolver(line, "") if resolver else line.strip()
            if resolved:
                _add(str(resolved))
    for path in (anchor.get("tapped") or []):
        _add(str(path))
    for value in (anchor.get("widgets") or {}).values():
        if not isinstance(value, str) or not value.strip():
            continue
        if Path(value.strip().strip('"')).suffix.lower() not in MEDIA_EXTS:
            continue  # a checkpoint / LoRA / sampler name is not a reference image
        resolved = resolver(value, "") if resolver else None
        if resolved:
            _add(str(resolved))
    return out


def briefing_from_hooks(hooks: list, resolver=None) -> QaBriefing | None:
    """Build a briefing from every ``purpose: "qa"`` hook on the canvas.

    Several QA hooks combine into one briefing (their criteria concatenate, their
    references union) rather than competing — the natural reading of two QA notes
    pinned to one graph is "both apply".
    """
    from src.utils.canvas_hooks import _is_qa

    criteria: list[str] = []
    refs: list[str] = []
    found = False
    for hook in (hooks or []):
        if not isinstance(hook, dict) or not _is_qa(hook):
            continue
        found = True
        text = str(hook.get("directive") or "").strip()
        if text:
            criteria.append(text)
        for anchor in (hook.get("anchors") or []):
            if isinstance(anchor, dict):
                for path in anchor_media_paths(anchor, resolver):
                    if path not in refs:
                        refs.append(path)
    if not found:
        return None
    body = "\n".join(criteria)
    budget, retry_hook = parse_retry(body)
    return QaBriefing(criteria=body, reference_paths=tuple(refs),
                      sources=("canvas qa hook",),
                      retry_budget=budget, retry_hook=retry_hook)


# ── surface 2: named briefing files ─────────────────────────────────────────────

# The folder documents itself; README.md is not a briefing anyone means to apply.
_NOT_BRIEFINGS = {"readme"}


def list_named_briefings() -> list[str]:
    """Names of the briefings on disk (``<briefing_dir>/<name>.md``), sorted."""
    d = briefing_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.md")
                  if p.is_file() and p.stem.lower() not in _NOT_BRIEFINGS)


def load_named_briefing(name: str) -> QaBriefing | None:
    """Load ``<briefing_dir>/<name>.md`` plus its optional ``<name>.refs/`` folder.

    The markdown is used verbatim as the criteria — it is the user's own writing,
    and reformatting someone's checklist is a good way to change what it means.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "", str(name or "")).strip(".")
    if not safe:
        return None
    path = briefing_dir() / f"{safe}.md"
    if not path.is_file():
        return None
    try:
        criteria = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("qa: could not read briefing %s — %s", path, exc)
        return None
    refs: list[str] = []
    refs_dir = briefing_dir() / f"{safe}.refs"
    if refs_dir.is_dir():
        for f in sorted(refs_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in MEDIA_EXTS:
                refs.append(str(f.resolve()))
    return QaBriefing(criteria=criteria, reference_paths=tuple(refs),
                      sources=(f"file:{safe}",))


def _expand_citations(briefing: QaBriefing) -> QaBriefing:
    """Fold in every ``@name`` a briefing cites, and strip the citation tokens.

    Lets a one-line canvas hook say "@house-style, plus keep the logo legible"
    without retyping a shared checklist.
    """
    names = [n for n in _CITE_RE.findall(briefing.criteria or "")]
    if not names:
        return briefing
    out = briefing
    resolved: list[str] = []
    for name in dict.fromkeys(names):
        cited = load_named_briefing(name)
        if cited is None:
            continue
        resolved.append(name)
        out = out.merged_with(cited)
    if not resolved:
        return out
    text = out.criteria
    for name in resolved:
        text = re.sub(rf"(?:^|\s)@{re.escape(name)}\b", " ", text)
    return QaBriefing(criteria=text.strip(), reference_paths=out.reference_paths,
                      sources=out.sources)


# ── surface 3: the per-thread /qa briefing ──────────────────────────────────────

def briefing_from_thread(thread_id: str) -> QaBriefing | None:
    """The briefing ``/qa`` stored against this conversation, if any."""
    if not thread_id:
        return None
    try:
        from src.utils import conversation_store as cs
        raw = cs.get_qa_briefing(thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not read the thread briefing: %s", exc)
        return None
    if not raw:
        return None
    refs = tuple(str(p) for p in (raw.get("reference_paths") or []))
    return QaBriefing(criteria=str(raw.get("criteria") or ""), reference_paths=refs,
                      sources=("/qa",))


# ── resolution ──────────────────────────────────────────────────────────────────

def resolve_briefing(hooks: list | None = None, thread_id: str = "",
                     resolver=None) -> QaBriefing | None:
    """The briefing in force for this turn, or None when QA should not run.

    A **canvas QA hook wins** over the thread's ``/qa`` briefing: it is the more
    specific, more visible statement — it's pinned to the graph the user is
    looking at. The thread briefing is the standing default for turns where no
    canvas says otherwise. Either may cite ``@name`` files, which are folded in.
    """
    if not qa_settings()["enabled"]:
        return None
    briefing = briefing_from_hooks(hooks or [], resolver)
    if briefing is None:
        briefing = briefing_from_thread(thread_id)
    if briefing is None:
        return None
    briefing = _expand_citations(briefing)
    return briefing if briefing else None


# ── the check ───────────────────────────────────────────────────────────────────

_VERDICT_KEYS = ("verdict", "checks", "summary")


@dataclass
class QaResult:
    """One judged output. ``passed`` is read from a structured field, never from
    the presence of the word FAIL in prose — the old checker did the latter and
    duly failed any verdict containing "no failures"."""
    path: str
    passed: bool
    summary: str = ""
    checks: list = field(default_factory=list)
    error: str = ""
    # True when nothing was judged because the QA model cannot read images at
    # all. `passed` is still True — our misconfiguration must not condemn the
    # user's work — but this says the pass means nothing, which a caller
    # deciding whether to re-render, or to claim an output was checked, needs.
    blind: bool = False

    def failed_criteria(self) -> list[str]:
        """The criteria that did not pass — what a retry has to fix."""
        out: list[str] = []
        for c in self.checks:
            if isinstance(c, dict) and str(c.get("result", "")).lower() not in ("pass", "n/a", "na", ""):
                text = str(c.get("criterion") or "").strip()
                note = str(c.get("note") or "").strip()
                out.append(f"{text} — {note}" if note and text else (text or note))
        return [t for t in out if t]

    def render(self) -> str:
        """A compact human line for the chat panel."""
        if self.error:
            return f"QA unavailable for `{Path(self.path).name}`: {self.error}"
        mark = "✅ PASS" if self.passed else "❌ FAIL"
        tail = f" — {self.summary}" if self.summary else ""
        return f"{mark} `{Path(self.path).name}`{tail}"


def parse_verdict(raw: str) -> dict:
    """Pull the verdict object out of a model reply.

    Tolerant on purpose: models wrap JSON in prose or fences even when told not
    to. Falls back to the first balanced ``{...}`` block. Returns {} when there is
    nothing usable, which the caller treats as "QA unavailable" rather than as a
    failure — an unreadable judge must never fail the user's work.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and any(k in data for k in _VERDICT_KEYS):
            return data
    return {}


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


def load_qa_prompts() -> dict[str, str]:
    """Parse the QA prompt file into its ``## <name>`` sections.

    The file holds both the agent's system prompt and the question templates the
    caller fills in; keeping them in one markdown file (rather than as strings in
    code) is what lets the prompts be edited without touching Python.
    """
    try:
        from src.utils.settings import load_settings
        filename = str((load_settings().get("system_prompts") or {})
                       .get("qa_checker", "system_prompt.qaChecker.md"))
    except Exception:  # noqa: BLE001
        filename = "system_prompt.qaChecker.md"
    if not filename.endswith(".md"):
        filename += ".md"
    config_dir = _PROJECT_ROOT / "config"
    path = config_dir / "system_prompts" / filename
    if not path.exists():
        path = config_dir / filename
    if not path.exists():
        logger.warning("qa: prompt file not found: %s", path)
        return {}
    text = path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+(.+)$", text, flags=re.MULTILINE)
    it = iter(parts[1:])  # parts[0] is whatever precedes the first heading
    for name, body in zip(it, it):
        sections[name.strip()] = body.strip()
    return sections


def _image_block(path: str) -> dict | None:
    """One Strands image ContentBlock, downsized to the provider's limits."""
    from src.tools.image_handling import _detect_format, _downsize, _MAX_IMAGE_BYTES

    try:
        raw = Path(path).read_bytes()
        fmt = _detect_format(path) or "png"
        data, fmt = _downsize(raw, fmt)
        if len(data) > _MAX_IMAGE_BYTES:
            logger.warning("qa: %s is still %d bytes after downsizing — skipped",
                           path, len(data))
            return None
        return {"image": {"format": fmt, "source": {"bytes": data}}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa: could not read %s — %s", path, exc)
        return None


# Ratios worth naming, as (label, width/height). A generated file rarely lands on
# an exact ratio (832x1472 reduces to 13:23, not 9:16), so the nearest of these
# within _RATIO_TOLERANCE is reported alongside the exact numbers.
_COMMON_RATIOS = [
    ("21:9", 21 / 9), ("2:1", 2.0), ("16:9", 16 / 9), ("3:2", 1.5), ("4:3", 4 / 3),
    ("5:4", 1.25), ("1:1", 1.0), ("4:5", 0.8), ("3:4", 0.75), ("2:3", 2 / 3),
    ("9:16", 9 / 16), ("1:2", 0.5), ("9:21", 9 / 21),
]
# Relative tolerance for claiming a common ratio. 832x1472 (0.5652 vs 9:16's
# 0.5625) is 0.5% off and is genuinely 9:16; 1000x437 is 2% off 21:9 and is not,
# so it stays reported as its own exact ratio rather than being rounded into one.
_RATIO_TOLERANCE = 0.015


def _describe_ratio(width: int, height: int) -> str:
    """Exact and nearest-common aspect ratio for *width* x *height*."""
    import math

    if not width or not height:
        return "unknown"
    value = width / height
    g = math.gcd(int(width), int(height))
    exact = f"{int(width) // g}:{int(height) // g}"
    nearest = min(_COMMON_RATIOS, key=lambda r: abs(r[1] - value) / r[1])
    close = abs(nearest[1] - value) / nearest[1] <= _RATIO_TOLERANCE
    orient = "square" if abs(value - 1) < 0.01 else ("landscape" if value > 1 else "portrait")
    if close and nearest[0] != exact:
        return f"{value:.3f} — {nearest[0]} ({orient}); exact pixel ratio {exact}"
    if close:
        return f"{value:.3f} — {nearest[0]} ({orient})"
    return f"{value:.3f} — {exact} ({orient}), not a standard ratio"


def measure_output(path: str) -> dict:
    """Hard, measured facts about a produced file.

    Vision models are reliably bad at exactly the properties that are trivial to
    compute — dimensions, aspect ratio, duration, frame count. Worse, the image
    they are shown has been resized on the way in, so their impression of its
    proportions is not evidence about the real file. A model asked to eyeball
    "is this 16:9?" will happily wave through a 9:16 render, which is precisely
    what it did before this existed. So we measure, and tell it not to guess.

    Returns {} when the file can't be read — the check then proceeds without
    facts rather than failing.
    """
    p = Path(path)
    facts: dict = {}
    try:
        facts["file"] = p.name
        facts["size_bytes"] = p.stat().st_size
    except OSError:
        return {}

    if is_image(path):
        try:
            from PIL import Image
            with Image.open(p) as img:
                facts.update({"width": img.width, "height": img.height,
                              "format": (img.format or "").upper(), "mode": img.mode})
        except Exception as exc:  # noqa: BLE001
            logger.debug("qa: could not measure image %s — %s", path, exc)
        return facts

    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(p))
        try:
            if cap.isOpened():
                facts["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                facts["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0:
                    facts["fps"] = round(fps, 3)
                if count > 0:
                    facts["frames"] = count
                if fps > 0 and count > 0:
                    facts["duration_s"] = round(count / fps, 2)
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not measure video %s — %s", path, exc)
    return facts


def render_measurements(facts: dict) -> str:
    """The measured facts as the block handed to the QA agent."""
    if not facts:
        return ""
    lines: list[str] = []
    w, h = facts.get("width"), facts.get("height")
    if w and h:
        lines.append(f"- dimensions: {w} x {h} px")
        lines.append(f"- aspect ratio: {_describe_ratio(w, h)}")
    if facts.get("duration_s") is not None:
        lines.append(f"- duration: {facts['duration_s']} s "
                     f"({facts.get('frames', '?')} frames @ {facts.get('fps', '?')} fps)")
    fmt = facts.get("format")
    size = facts.get("size_bytes")
    detail = ", ".join(str(x) for x in (fmt, facts.get("mode")) if x)
    if size:
        detail = (detail + ", " if detail else "") + f"{size / 1024 / 1024:.2f} MB"
    if detail:
        lines.append(f"- file: {detail}")
    return "\n".join(lines)


def _output_blocks(path: str, frames: int) -> tuple[list[dict], str]:
    """Image blocks for one produced file, plus how to describe them to the model.

    A video is sampled into evenly-spaced frames sent as ONE labelled sequence, so
    the model judges the clip — continuity, drift, a defect that appears halfway —
    rather than answering the same question about N unrelated stills, which is what
    the previous per-frame loop did.
    """
    if is_image(path):
        block = _image_block(path)
        return ([block] if block else []), "the GENERATED OUTPUT image"
    try:
        from agenty_core.utils.video_frames import extract_frames
        sampled = extract_frames(Path(path), count=max(1, frames))
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa: could not sample %s — %s", path, exc)
        return [], ""
    blocks = [b for b in (_image_block(str(f)) for f in (sampled or [])) if b]
    if not blocks:
        return [], ""
    return blocks, (f"{len(blocks)} FRAMES of the GENERATED OUTPUT video, in "
                    f"chronological order — judge them as one clip")


def check_output(path: str, briefing: QaBriefing, *, request: str = "",
                 agent=None) -> QaResult:
    """Judge one produced file against *briefing*.

    Never raises and never fails on doubt: if the model is unreachable or its reply
    can't be parsed, the result carries an ``error`` and counts as a PASS. A judge
    that can't be read must not be able to condemn the user's work — or worse,
    trigger a re-render loop on its own malfunction.
    """
    cfg = qa_settings()
    try:
        if agent is None:
            from src.agent import create_qa_agent
            agent = create_qa_agent()

        out_blocks, out_desc = _output_blocks(path, cfg["video_frames"])
        if not out_blocks:
            return QaResult(path=path, passed=True, error="output could not be read as an image/video")

        ref_paths = [p for p in briefing.reference_paths if is_image(p)][:cfg["max_references"]]
        ref_blocks = [b for b in (_image_block(p) for p in ref_paths) if b]

        # Say plainly that this is one output on its own. A run commonly makes
        # several and each is judged separately, so a criterion about the SET has
        # nothing here to judge — without this the model reports what it honestly
        # sees ("only one image") as a failure, and the whole batch is re-generated
        # for a reason no re-generation can address.
        alone = (" This is ONE output, judged on its own; the run may have produced "
                 "others you cannot see, so any criterion about how outputs compare "
                 "to EACH OTHER is n/a here.")
        if ref_blocks:
            n = len(ref_blocks)
            labels = ", ".join(f"IMAGE {i + 1}" for i in range(n))
            description = (f"You are given {n + len(out_blocks)} images: {labels} "
                           f"{'is a REFERENCE image' if n == 1 else 'are REFERENCE images'} "
                           f"from the user's briefing, then {out_desc}." + alone)
        else:
            description = f"You are given {out_desc}." + alone

        prompts = load_qa_prompts()
        criteria = briefing.criteria.strip() or prompts.get("no_criteria", "")
        measured = render_measurements(measure_output(path))
        question = (prompts.get("question", "")
                    .replace("{{IMAGE_DESCRIPTION}}", description)
                    .replace("{{REQUEST}}", (request or "").strip() or "(not recorded)")
                    .replace("{{MEASURED}}", measured or "(could not be measured)")
                    .replace("{{CRITERIA}}", criteria))
        if not question.strip():
            return QaResult(path=path, passed=True, error="QA prompt file has no `question` section")

        agent.messages.clear()  # stateless: this output is judged on its own
        reply = str(agent(ref_blocks + out_blocks + [{"text": question}]))
    except Exception as exc:  # noqa: BLE001
        # Passing on doubt is right for a judge that could not be REACHED. It is
        # wrong for one that cannot see: that is not doubt, it is a setting, and
        # it will wave through every output ever judged until someone changes it.
        # Still not a FAIL — condemning the user's work over our own
        # misconfiguration is the worse error — but it must not be silent.
        from src.utils.vision_capability import (blind_model_message, looks_blind,
                                                   model_name)
        if looks_blind(exc):
            model = model_name(agent)
            note = blind_model_message("qa_judge", model, str(exc))
            logger.error("qa: the QA model cannot see images — every output will "
                         "pass unchecked until this is changed (%s)", model or "?")
            try:
                from src.utils.status_bus import emit as _status
                _status("⚠️ QA is not actually checking anything: the qa_judge "
                        f"model{f' ({model})' if model else ''} cannot read images. "
                        "Point it at a vision model.")
            except Exception:  # noqa: BLE001
                pass
            return QaResult(path=path, passed=True, error=note, blind=True)
        logger.warning("qa: check failed for %s — %s", path, exc)
        return QaResult(path=path, passed=True, error=str(exc))

    data = parse_verdict(reply)
    if not data:
        logger.warning("qa: unparseable verdict for %s: %s", path, reply[:200])
        return QaResult(path=path, passed=True, error="the QA model returned no usable verdict")
    checks = [c for c in (data.get("checks") or []) if isinstance(c, dict)]
    verdict = str(data.get("verdict", "")).strip().lower()
    # Trust `verdict`, but a stated pass alongside a failed check is a contradiction
    # the user cares about — resolve it the safe way.
    failed = any(str(c.get("result", "")).strip().lower() == "fail" for c in checks)
    passed = (verdict == "pass") and not failed
    if verdict not in ("pass", "fail"):
        passed = not failed
    return QaResult(path=path, passed=passed, summary=str(data.get("summary") or "").strip(),
                    checks=checks)


def check_set(paths: list, briefing: QaBriefing, *, request: str = "",
              agent=None) -> QaResult:
    """Judge a whole run's outputs TOGETHER, for the criteria only a set can answer.

    Per-file QA cannot see a set, so "all the references must share one grade",
    "no two shots may repeat the same framing" and "the characters must stay
    consistent" are unjudgeable there — the per-file judge is told to mark them
    ``n/a`` precisely so it stops failing images for the absence of images it was
    never shown. This is where those criteria are actually answered.

    Same contract as :func:`check_output`: never raises, and an unreadable judge
    counts as a pass. ``path`` on the result carries the count rather than a file,
    since the verdict is about the set.
    """
    cfg = qa_settings()
    files = [str(p) for p in (paths or []) if p][:cfg["max_outputs"]]
    label = f"{len(files)} outputs"
    if len(files) < 2:
        return QaResult(path=label, passed=True,
                        error="a set verdict needs at least two outputs")
    try:
        if agent is None:
            from src.agent import create_qa_agent
            agent = create_qa_agent()

        out_blocks: list = []
        for i, p in enumerate(files):
            blocks, _desc = _output_blocks(p, 1)      # one frame each: this is about the set
            if blocks:
                out_blocks.extend(blocks[:1])
        if len(out_blocks) < 2:
            return QaResult(path=label, passed=True,
                            error="fewer than two outputs could be read")

        ref_paths = [p for p in briefing.reference_paths if is_image(p)][:cfg["max_references"]]
        ref_blocks = [b for b in (_image_block(p) for p in ref_paths) if b]
        n = len(ref_blocks)
        names = ", ".join(f"OUTPUT {i + 1} (`{Path(p).name}`)" for i, p in enumerate(files))
        description = (
            (f"You are given {n} REFERENCE image(s) from the user's briefing, then "
             if n else "You are given ")
            + f"ALL {len(out_blocks)} outputs of one run, in order: {names}. "
              "Judge them AS A SET: only the criteria that are about how the outputs "
              "relate to each other (consistency of style, grade, character identity, "
              "variety, no accidental repeats). A criterion about a single image on "
              "its own was already judged elsewhere — mark it n/a here.")

        prompts = load_qa_prompts()
        criteria = briefing.criteria.strip() or prompts.get("no_criteria", "")
        question = (prompts.get("question", "")
                    .replace("{{IMAGE_DESCRIPTION}}", description)
                    .replace("{{REQUEST}}", (request or "").strip() or "(not recorded)")
                    .replace("{{MEASURED}}", "(not applicable to a set)")
                    .replace("{{CRITERIA}}", criteria))
        if not question.strip():
            return QaResult(path=label, passed=True,
                            error="QA prompt file has no `question` section")

        agent.messages.clear()
        reply = str(agent(ref_blocks + out_blocks + [{"text": question}]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa: set check failed (%d outputs) — %s", len(files), exc)
        return QaResult(path=label, passed=True, error=str(exc))

    data = parse_verdict(reply)
    if not data:
        return QaResult(path=label, passed=True,
                        error="the QA model returned no usable verdict")
    checks = [c for c in (data.get("checks") or []) if isinstance(c, dict)]
    failed = any(str(c.get("result", "")).strip().lower() == "fail" for c in checks)
    verdict = str(data.get("verdict", "")).strip().lower()
    passed = (verdict == "pass") and not failed
    if verdict not in ("pass", "fail"):
        passed = not failed
    return QaResult(path=label, passed=passed,
                    summary=str(data.get("summary") or "").strip(), checks=checks)
