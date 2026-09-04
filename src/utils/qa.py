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
    # `forced_off` separates "the settings file says no" from "the environment
    # says no". A live canvas QA node overrides the first (see `resolve_briefing`)
    # — it is a deliberate per-graph act and beats a standing default — but not
    # the second, which stays an absolute kill switch for a CI or cost-capped run
    # that must not spend a judge's tokens whatever the canvas asks for.
    forced_off = False
    env = os.environ.get("AGENTY_QA")
    if env is not None and env.strip() != "":
        enabled = env.strip().lower() not in ("0", "false", "no", "off")
        forced_off = not enabled
    return {
        "enabled": enabled,
        "forced_off": forced_off,
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
    # Technical requirements set with checkboxes rather than written out — see
    # :mod:`src.utils.qa_checks`. They are settled by measuring the file, so they
    # never reach the model as something to judge.
    technical: dict = field(default_factory=dict)
    # Files named directly by an `agentY qa` node's `judge` input — a collector's
    # list, a loader's image, a path. Judged IN ADDITION to what the run produced,
    # never instead of it: `judge` says which outputs a briefing is about, and
    # reading it as "only these" would let a mis-wire quietly excuse everything
    # else from being checked.
    judge_paths: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.criteria.strip() or self.reference_paths or self.technical)

    def outputs_with(self, produced) -> list[str]:
        """What this briefing judges: the run's outputs plus any file it names.

        Order and de-duplication matter — the same file can arrive both ways (a
        collector wired into `judge` that holds this very run's output), and
        judging it twice would report one failure as two.
        """
        seen, out = set(), []
        for path in list(produced or []) + list(self.judge_paths):
            key = str(path)
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def merged_with(self, other: "QaBriefing | None") -> "QaBriefing":
        """This briefing with *other* folded in (other's criteria appended)."""
        if not other:
            return self
        criteria = "\n".join(t for t in (self.criteria.strip(), other.criteria.strip()) if t)
        refs = list(self.reference_paths)
        refs += [p for p in other.reference_paths if p not in set(refs)]
        judged = list(self.judge_paths)
        judged += [p for p in other.judge_paths if p not in set(judged)]
        # Later technical settings lose to earlier ones, matching `retry_budget`
        # above: the briefing nearest the work wins.
        technical = dict(other.technical or {})
        technical.update(self.technical or {})
        return QaBriefing(criteria=criteria, reference_paths=tuple(refs),
                          sources=tuple(dict.fromkeys(self.sources + other.sources)),
                          retry_budget=(self.retry_budget if self.retry_budget is not None
                                        else other.retry_budget),
                          retry_hook=self.retry_hook or other.retry_hook,
                          technical=technical, judge_paths=tuple(judged))

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
        tech = len([k for k, v in (self.technical or {}).items()
                    if v not in ("", "any", "off", None, False)])
        tech_txt = f", {tech} technical check{'' if tech == 1 else 's'}" if tech else ""
        return (f"{bullets} criteri{'on' if bullets == 1 else 'a'}{ref_txt}{tech_txt}"
                f"{retry} (from {where})")


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
    judged: list[str] = []
    technical: dict = {}
    retries: int | None = None
    found = False
    for hook in (hooks or []):
        if not isinstance(hook, dict) or not _is_qa(hook):
            continue
        found = True
        text = str(hook.get("directive") or "").strip()
        if text:
            criteria.append(text)
        # The `agentY qa briefing` node's dropdowns and switches. It arrives here
        # as a qa hook because that is what it is; what is different is that these
        # are settled by measuring the file rather than read by the model.
        spec = hook.get("technical")
        if isinstance(spec, dict):
            technical.update({k: v for k, v in spec.items()
                              if v not in ("", "any", None, False)})
        if retries is None and str(hook.get("retries", "")).strip().isdigit():
            retries = int(hook["retries"])
        for anchor in (hook.get("anchors") or []):
            if isinstance(anchor, dict):
                for path in anchor_media_paths(anchor, resolver):
                    if path not in refs:
                        refs.append(path)
        # `judged` are the nodes wired into an `agentY qa` node's `judge` input
        # that name files rather than a stage — a collector, a loader, a path.
        # Resolved through the same code as a reference, because the question
        # ("which files does this node mean?") is identical; what differs is which
        # side of the comparison they land on.
        for target in (hook.get("judged") or []):
            if isinstance(target, dict):
                for path in anchor_media_paths(target, resolver):
                    if path not in judged:
                        judged.append(path)
    if not found:
        return None
    body = "\n".join(criteria)
    # The technical requirements go into the criteria too. They are not judged
    # from there — they are already settled — but the briefing is also what the
    # user reads back, and a requirement appearing nowhere in it looks dropped.
    if technical:
        from src.utils.qa_checks import describe
        spoken = describe(technical)
        if spoken:
            body = (body + "\n" if body else "") + spoken
    budget, retry_hook = parse_retry(body)
    if budget is None and retries is not None:
        budget = retries
    return QaBriefing(criteria=body, reference_paths=tuple(refs),
                      sources=("canvas qa hook",),
                      retry_budget=budget, retry_hook=retry_hook,
                      technical=technical, judge_paths=tuple(judged))


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

def briefing_for_hook(hooks: list | None, hook_id: str,
                      resolver=None) -> QaBriefing | None:
    """The briefing in force for the outputs of ONE hook.

    A briefing left unwired applies to everything the run produces — right for a
    one-stage graph, wrong for a chain, where the reference frames and the video
    they feed want different standards. Wiring a briefing node's ``out`` into a
    hook's anchor scopes it to that hook, and the panel reports the link as
    ``applies_to``.

    So: unscoped briefings always count, a briefing naming *this* hook counts, and
    one naming a different hook is left out. Scoped ones are merged last, which is
    what makes them win a disagreement — the statement about this stage in
    particular beats the one about the graph in general.

    Used where the stage is known, which is the inline run path. The end-of-turn
    queued path has no hook to hand and keeps merging everything, so scoping can
    only ever narrow what a stage is judged against, never leave an output
    unchecked.
    """
    hid = str(hook_id or "")
    unscoped, scoped = [], []
    for hook in (hooks or []):
        if not isinstance(hook, dict):
            continue
        names = [str(x) for x in (hook.get("applies_to") or []) if str(x).strip()]
        if not names:
            unscoped.append(hook)
        elif hid and hid in names:
            scoped.append(hook)
    return briefing_from_hooks(unscoped + scoped, resolver)


def resolve_briefing(hooks: list | None = None, thread_id: str = "",
                     resolver=None) -> QaBriefing | None:
    """The briefing in force for this turn, or None when QA should not run.

    A **canvas QA hook wins** over the thread's ``/qa`` briefing: it is the more
    specific, more visible statement — it's pinned to the graph the user is
    looking at. The thread briefing is the standing default for turns where no
    canvas says otherwise. Either may cite ``@name`` files, which are folded in.

    For the same reason a canvas hook also **wins over ``qa.enabled``**. Wiring a
    QA node into the graph in front of you and leaving it live is a decision about
    THIS run; the settings switch is a standing default about runs in general, and
    a default that silently discards the specific instruction is the bug this
    exists to prevent. Bypassing or muting the node (Ctrl+B / Ctrl+M) is how you
    take it back — a disabled hook is never collected by the panel, so it never
    reaches here. ``AGENTY_QA=0`` in the environment still overrides everything.
    """
    cfg = qa_settings()
    briefing = briefing_from_hooks(hooks or [], resolver)
    if briefing is not None:
        if cfg.get("forced_off", False):
            return None
    else:
        if not cfg["enabled"]:
            return None
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
    # True when the judge could not read the image itself and was given a vision
    # agent's written description instead. The verdict counts — it is a real
    # check against the briefing — but it is second-hand, and a caller reporting
    # "this was checked" should say how.
    secondhand: bool = False

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
        via = " _(judged from a vision agent's description)_" if self.secondhand else ""
        return f"{mark} `{Path(self.path).name}`{tail}{via}"


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
        facts.update(_quality_facts(path, is_video=False))
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
    facts.update(_quality_facts(path, is_video=True))
    return facts


def _quality_facts(path: str, *, is_video: bool) -> dict:
    """Sharpness, noise, exposure — the same trade as the dimensions above.

    Softness, grain and blown highlights are the complaints people actually make,
    and a vision model estimates all three badly from a resized copy. Measuring
    them costs milliseconds and no GPU, so the judge is handed numbers instead of
    an impression. Never fatal: an unreadable frame simply contributes nothing.
    """
    try:
        from src.utils.image_facts import measure
        return measure(path, is_video=is_video)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not measure quality of %s — %s", path, exc)
        return {}


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
    try:
        from src.utils.image_facts import render_quality
        lines.extend(render_quality(facts))
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not render quality facts — %s", exc)
    try:
        from src.utils.likeness import render_likeness
        lines.extend(render_likeness(facts))
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not render likeness facts — %s", exc)
    # The one number that is NOT a gate. It goes in labelled as a ranking aid,
    # because the judge is being handed a pile of thresholds either side of it
    # and would otherwise be entitled to read this as one more of them.
    try:
        from src.utils.fitness import render_score, score
        line = render_score(score(facts))
        if line:
            lines.append(line)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not render the quality score — %s", exc)
    return "\n".join(lines)


def _output_frame_paths(path: str, frames: int) -> tuple[list[str], str]:
    """The files that stand for one produced output, plus how to describe them.

    Separate from :func:`_output_blocks` because a judge that cannot read images
    needs the same files as *paths*, to hand to the vision agent — see
    :func:`_judge_from_descriptions`.
    """
    if is_image(path):
        return [path], "the GENERATED OUTPUT image"
    try:
        from agenty_core.utils.video_frames import extract_frames
        sampled = extract_frames(Path(path), count=max(1, frames))
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa: could not sample %s — %s", path, exc)
        return [], ""
    paths = [str(f) for f in (sampled or [])]
    if not paths:
        return [], ""
    return paths, (f"{len(paths)} FRAMES of the GENERATED OUTPUT video, in "
                   f"chronological order — judge them as one clip")


def _output_blocks(path: str, frames: int) -> tuple[list[dict], str]:
    """Image blocks for one produced file, plus how to describe them to the model.

    A video is sampled into evenly-spaced frames sent as ONE labelled sequence, so
    the model judges the clip — continuity, drift, a defect that appears halfway —
    rather than answering the same question about N unrelated stills, which is what
    the previous per-frame loop did.
    """
    paths, desc = _output_frame_paths(path, frames)
    blocks = [b for b in (_image_block(p) for p in paths) if b]
    if not blocks:
        return [], ""
    return blocks, desc


def _judge_question(path: str, briefing: QaBriefing, request: str,
                    description: str, ref_paths: list) -> tuple[str, list[dict]]:
    """The question put to the judge, plus the checks arithmetic already settled.

    *description* is the "You are given …" sentence, which differs between the two
    ways this gets asked: pixels attached, or a vision agent's written description
    of them (:func:`_judge_from_descriptions`). Everything else — the measured
    facts, the settled technical checks, the criteria — is the same question
    either way, and is built here once so the two paths cannot drift apart.
    """
    prompts = load_qa_prompts()
    criteria = briefing.criteria.strip() or prompts.get("no_criteria", "")
    facts = measure_output(path)
    facts.update(_likeness_facts(path, briefing, ref_paths))
    measured = render_measurements(facts)
    # The technical half is decided here, by arithmetic, before the model is
    # asked anything. It is then shown the answers so it does not guess at
    # the same questions and contradict them.
    settled = _settle_technical(briefing, facts)
    if settled:
        from src.utils.qa_checks import render_for_model
        measured = (measured + "\n\n" + render_for_model(settled)).strip()
    question = (prompts.get("question", "")
                .replace("{{IMAGE_DESCRIPTION}}", description)
                .replace("{{REQUEST}}", (request or "").strip() or "(not recorded)")
                .replace("{{MEASURED}}", measured or "(could not be measured)")
                .replace("{{CRITERIA}}", criteria))
    return question, settled


def _vision_describe(paths: list, question: str) -> list[str]:
    """Ask the vision agent to describe *paths*, the way the orchestrator does.

    Same route as :func:`src.tools.image_handling.analyze_image` in ``describe``
    mode — the pixels go to the vision agent and only text comes back — so a QA
    model that cannot take image input can still be asked about an image.

    Returns one description per file it managed to read, or ``[]`` when there is
    no vision agent registered, or when the vision model turns out to be blind
    too. Never raises: this runs inside a failure handler.
    """
    try:
        from src.tools import image_handling as _ih
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa: no image tooling to relay to (%s)", exc)
        return []
    # Ask directly rather than reading the answer out of a fallback: with no
    # vision agent registered `analyze_image` quietly degrades to returning raw
    # bytes, which is exactly what the caller here cannot use.
    if getattr(_ih, "_vision_agent", None) is None:
        logger.warning("qa: the QA model cannot see and no vision agent is "
                       "registered to relay to")
        return []
    out: list[str] = []
    for p in paths:
        try:
            res = _ih.analyze_image(file_path=str(p), question=question,
                                    mode="describe") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("qa: vision relay failed for %s — %s", p, exc)
            return []
        if str(res.get("status", "")).lower() not in ("success", "ok"):
            # A blind vision model reports itself here. Relaying to a second model
            # that also cannot see is not a fallback, it is the same failure.
            logger.warning("qa: vision relay unavailable for %s — %s", p,
                           str(res.get("content"))[:200])
            return []
        text = " ".join(str(b.get("text") or "") for b in (res.get("content") or [])
                        if isinstance(b, dict)).strip()
        if text:
            out.append(text)
    return out


def _judge_from_descriptions(path: str, briefing: QaBriefing, request: str,
                             cfg: dict, agent) -> tuple[str, list[dict]] | None:
    """Judge one output through the vision agent, for a QA model that cannot see.

    The orchestrator has always handled its own blindness this way: it never
    looks at pixels, it asks the vision agent and reasons over the description.
    A text-only ``qa_judge`` can do exactly the same, and the alternative it
    replaces is worthless — a blind judge passes every output forever.

    A verdict from someone else's description is weaker than one from the image,
    so the caller marks it: see ``QaResult.secondhand``. Returns None when the
    relay is not available, leaving the caller to report the blindness.
    """
    out_paths, out_desc = _output_frame_paths(path, cfg["video_frames"])
    if not out_paths:
        return None
    # Point the describer at what actually has to be decided. A generic
    # "describe this image" spends its words on composition and mood and omits
    # the one detail the briefing turns on.
    ask = ("Describe this image in detail for a quality check. Be concrete and "
           "literal about the subject, and cover these points specifically:\n"
           + (briefing.criteria.strip() or "overall quality and content"))
    out_texts = _vision_describe(out_paths, ask)
    if not out_texts:
        return None
    ref_paths = [p for p in briefing.reference_paths if is_image(p)][:cfg["max_references"]]
    ref_texts = _vision_describe(ref_paths, ask) if ref_paths else []

    alone = (" This is ONE output, judged on its own; the run may have produced "
             "others you cannot see, so any criterion about how outputs compare "
             "to EACH OTHER is n/a here.")
    lines = ["You cannot see images, so a vision model looked at them for you and "
             "wrote the descriptions below. Judge from these descriptions."]
    for i, text in enumerate(ref_texts, 1):
        lines.append(f"\nDESCRIPTION OF REFERENCE IMAGE {i}:\n{text}")
    label = ("DESCRIPTION OF " + out_desc.upper()) if out_desc else "DESCRIPTION OF THE OUTPUT"
    for i, text in enumerate(out_texts, 1):
        suffix = f" ({i} of {len(out_texts)})" if len(out_texts) > 1 else ""
        lines.append(f"\n{label}{suffix}:\n{text}")
    # A description cannot answer everything, and a judge that guesses anyway is
    # the failure mode this whole path exists to avoid.
    lines.append("\nA description is second-hand evidence: mark a criterion `n/a` "
                 "when the description does not settle it, rather than guessing."
                 + alone)
    description = "\n".join(lines)

    question, settled = _judge_question(path, briefing, request, description, ref_paths)
    if not question.strip():
        return None
    agent.messages.clear()
    reply = str(agent([{"text": description + "\n\n" + question}]))
    return reply, settled


def _set_from_descriptions(files: list, briefing: QaBriefing, request: str,
                           cfg: dict, agent) -> str | None:
    """The set question, asked through the vision agent, for a blind QA model.

    One description per output — the set criteria are about how they relate, so
    each needs saying once and the comparison happens in the judge's reading of
    them. Returns None when there is nothing to relay through.
    """
    ask = ("Describe this image in detail for a quality check, so it can be "
           "compared against others from the same run. Be concrete about colour "
           "grade, lighting, style and subject, and cover:\n"
           + (briefing.criteria.strip() or "overall quality and content"))
    texts: list[str] = []
    for p in files:
        got = _vision_describe([str(p)], ask)
        if not got:
            return None
        texts.append(got[0])
    if len(texts) < 2:
        return None
    ref_paths = [p for p in briefing.reference_paths if is_image(p)][:cfg["max_references"]]
    ref_texts = _vision_describe(ref_paths, ask) if ref_paths else []

    lines = ["You cannot see images, so a vision model looked at them for you and "
             "wrote the descriptions below. Judge from these descriptions."]
    for i, text in enumerate(ref_texts, 1):
        lines.append(f"\nDESCRIPTION OF REFERENCE IMAGE {i}:\n{text}")
    for i, (p, text) in enumerate(zip(files, texts), 1):
        lines.append(f"\nDESCRIPTION OF OUTPUT {i} (`{Path(str(p)).name}`):\n{text}")
    lines.append("\nJudge them AS A SET: only the criteria about how the outputs "
                 "relate to each other (consistency of style, grade, character "
                 "identity, variety, no accidental repeats). A criterion about a "
                 "single image on its own was already judged elsewhere — mark it "
                 "n/a here. A description is second-hand evidence: mark a "
                 "criterion `n/a` when the descriptions do not settle it, rather "
                 "than guessing.")
    description = "\n".join(lines)

    prompts = load_qa_prompts()
    criteria = briefing.criteria.strip() or prompts.get("no_criteria", "")
    question = (prompts.get("question", "")
                .replace("{{IMAGE_DESCRIPTION}}", description)
                .replace("{{REQUEST}}", (request or "").strip() or "(not recorded)")
                .replace("{{MEASURED}}", "(not applicable to a set)")
                .replace("{{CRITERIA}}", criteria))
    if not question.strip():
        return None
    agent.messages.clear()
    return str(agent([{"text": description + "\n\n" + question}]))


def check_output(path: str, briefing: QaBriefing, *, request: str = "",
                 agent=None) -> QaResult:
    """Judge one produced file against *briefing*.

    Never raises and never fails on doubt: if the model is unreachable or its reply
    can't be parsed, the result carries an ``error`` and counts as a PASS. A judge
    that can't be read must not be able to condemn the user's work — or worse,
    trigger a re-render loop on its own malfunction.
    """
    cfg = qa_settings()
    settled: list[dict] = []
    reply: str | None = None
    secondhand = False
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

        question, settled = _judge_question(path, briefing, request,
                                            description, ref_paths)
        if not question.strip():
            return QaResult(path=path, passed=True, error="QA prompt file has no `question` section")

        agent.messages.clear()  # stateless: this output is judged on its own
        reply = str(agent(ref_blocks + out_blocks + [{"text": question}]))
    except Exception as exc:  # noqa: BLE001
        # Passing on doubt is right for a judge that could not be REACHED. It is
        # wrong for one that cannot see: that is not doubt, it is a setting, and
        # it will wave through every output ever judged until someone changes it.
        from src.utils.vision_capability import (blind_model_message, looks_blind,
                                                   model_name)
        blind = looks_blind(exc)
        if blind:
            # Blind is not the end of it. The orchestrator has never read pixels
            # either and works fine, because it asks the vision agent — so ask
            # the vision agent. A verdict from a description is weaker than one
            # from the image, never worse than the nothing it replaces.
            try:
                relayed = _judge_from_descriptions(path, briefing, request, cfg, agent)
            except Exception as relay_exc:  # noqa: BLE001
                logger.warning("qa: vision relay failed for %s — %s", path, relay_exc)
                relayed = None
            if relayed is not None:
                reply, settled = relayed
                secondhand = True
        if reply is None and blind:
            # Nothing to relay through either. Still not a FAIL — condemning the
            # user's work over our own misconfiguration is the worse error — but
            # it must not be silent.
            model = model_name(agent)
            note = blind_model_message("qa_judge", model, str(exc))
            logger.error("qa: the QA model cannot see images and no vision agent "
                         "could stand in — every output will pass unchecked until "
                         "this is changed (%s)", model or "?")
            try:
                from src.utils.status_bus import emit as _status
                _status("⚠️ QA is not actually checking anything: the qa_judge "
                        f"model{f' ({model})' if model else ''} cannot read images "
                        "and no vision agent was available to look for it. Point "
                        "qa_judge at a vision model.")
            except Exception:  # noqa: BLE001
                pass
            return QaResult(path=path, passed=True, error=note, blind=True)
        if reply is None:
            logger.warning("qa: check failed for %s — %s", path, exc)
            return QaResult(path=path, passed=True, error=str(exc))

    data = parse_verdict(reply)
    if not data:
        logger.warning("qa: unparseable verdict for %s: %s", path, reply[:200])
        return QaResult(path=path, passed=True, error="the QA model returned no usable verdict")
    model_checks = [c for c in (data.get("checks") or []) if isinstance(c, dict)]
    # The measured verdicts go in FIRST and are not overwritten: a model that
    # re-judged one anyway must not be able to talk its way past a number.
    named = {str(s.get("criterion", "")).strip().lower() for s in settled}
    checks = settled + [c for c in model_checks
                        if str(c.get("criterion", "")).strip().lower() not in named]
    verdict = str(data.get("verdict", "")).strip().lower()
    # Trust `verdict`, but a stated pass alongside a failed check is a contradiction
    # the user cares about — resolve it the safe way.
    failed = any(str(c.get("result", "")).strip().lower() == "fail" for c in checks)
    passed = (verdict == "pass") and not failed
    if verdict not in ("pass", "fail"):
        passed = not failed
    return QaResult(path=path, passed=passed, summary=str(data.get("summary") or "").strip(),
                    checks=checks, secondhand=secondhand)


# How many frames of a clip are compared against the references. A character can
# be out of shot for part of a take, so one frame is not enough; every extra one
# costs a full comparison against every reference, so it is not many either.
LIKENESS_FRAMES = 3


def _likeness_facts(path: str, briefing: QaBriefing, ref_paths: list) -> dict:
    """How much the output looks like the references, as a number.

    Only computed when the briefing's likeness control actually asked for it. The
    subject scorer's first load is ~100 s and 3 GB of weights, and a run that
    never mentions a reference must not pay a second of that — so the question is
    checked before anything is imported.

    Never raises. An unmeasurable comparison leaves the written criterion to reach
    the model on its own, which is a worse answer but always an available one.
    """
    want = str((getattr(briefing, "technical", None) or {}).get("likeness") or "")
    refs = [str(p) for p in (ref_paths or [])]
    if not want or not refs:
        return {}
    try:
        from src.utils.qa_checks import LIKENESS_SCORERS
        key = LIKENESS_SCORERS.get(want)
        if not key:
            return {}
        candidates = [path] if is_image(path) else _likeness_frames(path)
        if not candidates:
            return {}
        import src.utils.likeness as likeness
        got = getattr(likeness, key)(candidates, refs)
        return {key: got} if got else {}
    except Exception as exc:  # noqa: BLE001
        logger.info("qa: could not measure likeness for %s — %s", path, exc)
        return {}


def _likeness_frames(path: str) -> list:
    """A clip as a handful of stills, so it can be compared to a reference."""
    try:
        from agenty_core.utils.video_frames import extract_frames
        return [str(f) for f in (extract_frames(Path(path), count=LIKENESS_FRAMES) or [])]
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: could not sample %s for likeness — %s", path, exc)
        return []


def _settle_technical(briefing: QaBriefing, facts: dict) -> list[dict]:
    """The briefing's checkbox requirements, judged against the measured file.

    Never raises: a technical check that cannot be evaluated contributes nothing,
    which leaves the written criteria to carry the verdict on their own.
    """
    if not getattr(briefing, "technical", None) or not facts:
        return []
    try:
        from src.utils.qa_checks import evaluate
        return evaluate(briefing.technical, facts)
    except Exception as exc:  # noqa: BLE001
        logger.debug("qa: technical checks failed — %s", exc)
        return []


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
    secondhand = False
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
        # Same relay as the per-file judge: a QA model that cannot take images
        # can still be asked about them through the vision agent. A set question
        # ("do these share one grade?") survives the trip surprisingly well —
        # it is exactly the kind of thing a description states outright.
        from src.utils.vision_capability import looks_blind
        reply = None
        if looks_blind(exc):
            try:
                reply = _set_from_descriptions(files, briefing, request, cfg, agent)
            except Exception as relay_exc:  # noqa: BLE001
                logger.warning("qa: set vision relay failed — %s", relay_exc)
                reply = None
            secondhand = reply is not None
        if reply is None:
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
    return QaResult(path=label, passed=passed, secondhand=secondhand,
                    summary=str(data.get("summary") or "").strip(), checks=checks)
