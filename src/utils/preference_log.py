"""What the user actually kept, and what they threw away, with the numbers.

The weights in :mod:`src.utils.fitness` are a guess. Turning them into something
better needs labels — someone saying *this one, not that one* — and asking for
those separately would be a chore nobody does twice.

But there is already a moment where exactly that happens, for its own reasons.
A `review` hook stops a chain, fills a collector with everything the stage
produced, and waits. The user deletes the rows they do not want and says
continue. What survives is what they chose; what they removed is what they
rejected. That is a **choice from a slate**: k kept out of N shown, produced as a side
effect of work they were doing anyway — the cheapest training data there is, and
the only kind that reflects this user's taste rather than a published benchmark's.

The slate is the unit, not the pair. Someone who keeps one image out of eight has
told you one thing about eight items, not seven independent things; see
:func:`slates` for what follows from that.

So every answered review is written here as one event, with each file's measured
feature vector alongside. The vectors matter more than the paths: outputs get
deleted, folders get moved, and a label whose evidence is gone is worthless.
Stored this way the log stays refittable years after the pictures are gone.

**Nothing here changes what runs.** Recording is best-effort and must never
raise, never block, and never touch the answer — a review is the user's decision,
and a logging failure has no business affecting it.

**A review where nothing was dropped is not recorded.** Keeping everything is
not a preference between anything, so it would add rows and no information.

The file is JSONL at ``output/agent/preferences.jsonl`` inside this checkout —
not beside the media in ComfyUI's output folder, because it has to outlive any
particular project's outputs and must not depend on ComfyUI being reachable to
be written. ``output/`` is gitignored: these are this machine's judgements, not
the repository's.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("agentY.preference_log")

LOG_PATH = (Path(__file__).resolve().parent.parent.parent
            / "output" / "agent" / "preferences.jsonl")

# A single review with a hundred outputs would write ten thousand implied pairs.
# The event is capped instead, at a size well past any real review.
MAX_PER_SIDE = 40


def _measure(path: str, facts_by_path: dict | None) -> dict:
    """The measured facts for one file — reused if the caller already has them."""
    if facts_by_path and path in facts_by_path:
        return facts_by_path[path] or {}
    try:
        from src.utils.image_facts import measure
        return measure(path, is_video=not _looks_image(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("preference_log: could not measure %s — %s", path, exc)
        return {}


def _looks_image(path: str) -> bool:
    return Path(str(path)).suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def _entry(path: str, facts_by_path: dict | None) -> dict | None:
    """One file as it is stored: its name, its features, and its score today.

    ``None`` when it has no features at all — a row that cannot be compared with
    anything is not a label, and writing it would only dilute the fit.
    """
    from src.utils.fitness import features, score
    facts = _measure(str(path), facts_by_path)
    feats = features(facts)
    if not feats:
        return None
    s = score(facts)
    return {
        "name": Path(str(path)).name,
        "path": str(path),
        "features": {k: round(float(v), 4) for k, v in feats.items()},
        # Informational: what the weights of the day made of it. The fit reads
        # `features`, never this — otherwise a refit would be learning from its
        # own previous answer.
        "score": s.get("score"),
    }


def record_review(kept, dropped, *, hook_node_id: str = "", question: str = "",
                  request: str = "", facts_by_path: dict | None = None,
                  path: Path | None = None) -> int:
    """Log one answered review. Returns the number of implied pairs, or 0.

    Never raises. *facts_by_path* lets a caller that has already measured these
    files hand the numbers over rather than paying for them twice.
    """
    try:
        keep = [str(p) for p in (kept or []) if p][:MAX_PER_SIDE]
        drop = [str(p) for p in (dropped or []) if p][:MAX_PER_SIDE]
        if not keep or not drop:
            return 0
        chosen = [e for e in (_entry(p, facts_by_path) for p in keep) if e]
        rejected = [e for e in (_entry(p, facts_by_path) for p in drop) if e]
        if not chosen or not rejected:
            return 0
        event = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "review",
            "hook_node_id": str(hook_node_id or ""),
            "question": str(question or "")[:400],
            "request": str(request or "")[:400],
            "chosen": chosen,
            "rejected": rejected,
        }
        target = Path(path) if path else LOG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return len(chosen) * len(rejected)
    except Exception as exc:  # noqa: BLE001 — a label is never worth a turn
        logger.debug("preference_log: could not record a review — %s", exc)
        return 0


def read_events(path: Path | None = None) -> list:
    """Every logged event. A corrupt line is skipped, not fatal."""
    target = Path(path) if path else LOG_PATH
    out: list = []
    try:
        text = target.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — no log yet is the normal case
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(row, dict) and row.get("chosen") and row.get("rejected"):
            out.append(row)
    return out


def slates(events=None, path: Path | None = None) -> list:
    """Every review as ``(chosen_features, rejected_features)`` — the SLATE form.

    This is the shape the fit wants. A review is not a bag of independent duels:
    the user saw all N outputs at once and kept k of them, so the thing that
    describes it is a **choice from a slate** (Plackett-Luce / Luce). Bradley-Terry
    is its two-item case, so a review that kept 1 of 2 produces exactly that.

    Decomposing a slate into duels is not *wrong* — it is a composite likelihood
    of the same model and, measured, just as accurate (see
    :mod:`src.utils.fitness_fit`). What it breaks is the counting: one review
    becomes seven observations, and every constant that means "per decision"
    quietly starts meaning "per duel". Keeping the slate whole keeps that honest.

    Only features present across the WHOLE slate survive: the score of every
    member goes into one shared denominator, so a feature that some members lack
    cannot take part in that comparison at all.
    """
    rows = events if events is not None else read_events(path)
    out: list = []
    for ev in (rows or []):
        chosen = [c.get("features") or {} for c in (ev.get("chosen") or [])]
        rejected = [r.get("features") or {} for r in (ev.get("rejected") or [])]
        if not chosen or not rejected:
            continue
        shared = set(chosen[0])
        for f in chosen[1:] + rejected:
            shared &= set(f)
        keys = sorted(shared)
        if not keys:
            continue
        out.append(([{k: float(f[k]) for k in keys} for f in chosen],
                    [{k: float(f[k]) for k in keys} for f in rejected]))
    return out


def pairs(events=None, path: Path | None = None) -> list:
    """Every implied preference as ``(winner_features, loser_features)``.

    Kept for *measuring* a fit rather than making one: "how often does this put a
    kept output above a rejected one" is the plainest accuracy there is, and it is
    read the same way whatever likelihood produced the weights.
    """
    out: list = []
    for chosen, rejected in slates(events, path):
        for win in chosen:
            for lose in rejected:
                out.append((dict(win), dict(lose)))
    return out


def summary(path: Path | None = None) -> str:
    """One line for whoever asks how much training data there is."""
    events = read_events(path)
    if not events:
        return "no preference labels recorded yet"
    usable = slates(events)
    n = len(pairs(events))
    return (f"{len(events)} review{'' if len(events) == 1 else 's'} logged, "
            f"{len(usable)} usable slate{'' if len(usable) == 1 else 's'}, "
            f"{n} implied preference pair{'' if n == 1 else 's'}")
