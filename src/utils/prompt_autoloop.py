"""The prompt loop's unsupervised mode: generate, judge, rewrite, until it passes.

The supervised loop (``revise_prompt``) writes a prompt, the panel queues it, the
user looks. This mode is asked for in chat ("keep going until it's right") and
takes the user out of it: the orchestrator writes each version, agentY runs the
graph itself (no browser in the loop), the QA agent judges the render against
the canvas QA node plus any goal the user stated, and the orchestrator reads the
verdict and writes the next version — all in one turn.

This module is the bookkeeping only, so it can be tested without a model or a
ComfyUI: what counts as circling, when to stop, which version was best, and the
summary. The round itself is ``Pipeline.prompt_autoloop``.

What it adds over running ``revise_prompt`` in a loop:

* **Keep the best.** The last version is not necessarily the best one — a
  revision that fixes one note can break a criterion that had passed. When the
  loop ends without a pass, the best version (fewest missed criteria, then the
  fitness score) is put back on the canvas.
* **Stop when it circles.** A criterion that fails twice running is re-rolled
  once with the same prompt and a fresh seed; that separates "the prompt cannot
  say it" from "that seed was unlucky". A third failure ends the loop — no
  rewording is reaching it.
* **Say what happened.** The summary names the winner, why, and what the judge
  kept objecting to.
"""

from __future__ import annotations

# A criterion failing this many rounds in a row gets one seed re-roll ...
REROLL_AT = 2
# ... and this many ends the loop: rewording and re-rolling have both missed it.
STALL_AT = 3


def new_session(budget: int, briefing, goal: str = "") -> dict:
    """A fresh unsupervised run: its budget, its judge, and no rounds yet."""
    return {"budget": int(budget), "briefing": briefing, "goal": str(goal or ""),
            "runs": [], "done": ""}


def criterion_key(missed: str) -> str:
    """The criterion a failure is about, without the judge's note on it.

    ``failed_criteria()`` gives "criterion — note"; the note differs every round
    even when the same thing is wrong, so a streak is counted on the criterion.
    """
    return str(missed or "").split(" — ")[0].strip().lower()


def streaks(runs: list) -> dict:
    """``{criterion: rounds it has failed in a row, up to the latest}``."""
    out: dict = {}
    if not runs:
        return out
    for key in {criterion_key(m) for m in (runs[-1].get("missed") or [])}:
        n = 0
        for run in reversed(runs):
            if key in {criterion_key(m) for m in (run.get("missed") or [])}:
                n += 1
            else:
                break
        out[key] = n
    return out


def decide(session: dict, *, interrupted: bool = False) -> tuple[str, str]:
    """What happens after the latest round: ``(next, why)``.

    ``next`` is ``revise`` or ``reroll`` to go on, or the reason it ended:
    ``passed``, ``unjudged``, ``budget``, ``stalled``, ``interrupted``.
    """
    runs = session.get("runs") or []
    if not runs:
        return "revise", ""
    last = runs[-1]
    if last.get("error"):
        # The judge passes on doubt so it can never condemn the user's work; here
        # the same doubt would end the loop in a success nobody checked.
        return "unjudged", f"the judge could not be read ({last['error']})"
    if last.get("passed"):
        return "passed", f"v{last['v']} passed QA"
    if interrupted:
        return "interrupted", "the user said something while it ran"
    worst = max(streaks(runs).items(), key=lambda kv: kv[1], default=("", 0))
    if worst[1] >= STALL_AT:
        return "stalled", (f"“{worst[0]}” failed {worst[1]} rounds running, through "
                           "rewording and a fresh seed — the prompt is not reaching it")
    if len(runs) >= int(session.get("budget") or 0):
        return "budget", f"all {len(runs)} runs used"
    if worst[1] >= REROLL_AT and not last.get("reroll"):
        return "reroll", (f"“{worst[0]}” failed {worst[1]} rounds running — re-rolling "
                          "the seed with the same prompt, to see whether it is the "
                          "prompt or the roll")
    return "revise", ""


def best(runs: list) -> dict | None:
    """The version to leave on the canvas.

    A pass beats everything; then the fewest missed criteria; then the fitness
    score (a render that could not be measured ranks below one that could);
    then the later round, which had the more informed prompt.
    """
    judged = [r for r in (runs or []) if r.get("output")]
    if not judged:
        return None

    def rank(r):
        score = r.get("score")
        return (bool(r.get("passed")), -len(r.get("missed") or []),
                -1.0 if score is None else float(score), int(r.get("v") or 0))
    return max(judged, key=rank)


def objections(runs: list, top: int = 3) -> list[str]:
    """The criteria the judge failed most often, most frequent first."""
    count: dict = {}
    for run in runs or []:
        for key in {criterion_key(m) for m in (run.get("missed") or [])}:
            count[key] = count.get(key, 0) + 1
    ranked = sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"{k} (failed {n}×)" for k, n in ranked[:top]]


def summary(session: dict, outcome: str, why: str) -> dict:
    """The end-of-run report the orchestrator relays to the user."""
    runs = session.get("runs") or []
    win = best(runs)
    rows = [{"v": r.get("v"), "passed": bool(r.get("passed")),
             "missed": len(r.get("missed") or []),
             "score": r.get("score"), "seed_reroll": bool(r.get("reroll")),
             "output": r.get("output", "")} for r in runs]
    out = {"outcome": outcome, "why": why, "runs": len(runs),
           "budget": session.get("budget"), "rounds": rows,
           "kept_objecting_to": objections(runs)}
    if win:
        out["best"] = {"v": win.get("v"), "passed": bool(win.get("passed")),
                       "missed": list(win.get("missed") or []),
                       "score": win.get("score"), "output": win.get("output", ""),
                       "is_last": win is runs[-1]}
    return out
