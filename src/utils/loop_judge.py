"""Judging one round of a canvas loop.

The loop's condition is the user's sentence on the ``agentY loop break`` node. It
is judged the way a QA briefing is — by the QA agent, on the file, with the
measurable parts measured — and together with the QA briefings that cover the
loop's stages, so "finished" means the condition AND the briefings.

The decision of what a verdict leads to is :mod:`src.utils.hook_flow`'s; this
module only produces the verdicts.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("agentY.loop")


def loop_briefing(loop, stage_briefings: list | None = None):
    """The condition as a briefing, with the stages' own briefings folded in."""
    from src.utils.qa import QaBriefing
    lines = [ln.strip() for ln in str(loop.condition or "").splitlines() if ln.strip()]
    briefing = QaBriefing(criteria="\n".join(lines), sources=(f"loop break {loop.break_id}",))
    for other in stage_briefings or []:
        if other:
            briefing = briefing.merged_with(other)
    return briefing


def judge(paths: list, loop, stage_briefings: list | None = None, *,
          request: str = "", check=None, score=None) -> list[dict]:
    """One verdict per file: ``{path, passed, missed, summary, score, error}``.

    *check* and *score* are the QA agent's ``check_output`` and the quality
    score's ``score_file``; passed in by tests.
    """
    briefing = loop_briefing(loop, stage_briefings)
    if check is None:
        from src.utils.qa import check_output as check
    if score is None:
        from src.utils.fitness import score_file as score
    results = []
    for path in [str(p) for p in (paths or []) if p]:
        entry = {"path": path, "passed": True, "missed": [], "summary": "", "score": 0.0, "error": ""}
        if briefing:
            try:
                verdict = check(path, briefing, request=request)
                entry.update(passed=bool(verdict.passed), missed=verdict.failed_criteria(),
                             summary=str(verdict.summary or ""), error=str(verdict.error or ""))
            except Exception as exc:  # noqa: BLE001 - a judge that fails must not condemn the work
                entry["error"] = f"{type(exc).__name__}: {exc}"
        try:
            entry["score"] = float((score(path) or {}).get("score") or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("loop: no quality score for %s (%s)", path, exc)
        results.append(entry)
    return results
