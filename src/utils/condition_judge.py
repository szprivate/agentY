"""The ``agentY condition`` node as the judge of a loop.

On a canvas the node answers a yes/no about a picture with a decision model
(clef / clef-flash, run by Ollama). Put on the execution wire inside a loop it
is what judges each round: the files the round produced are asked the node's
statements, and the loop ends when one of them meets them all.

The rules are the node's own (``agentY-comfyuiConnect/_condition.py``) and must
stay so - a picture that passes on the canvas has to pass here: one statement per
line, ``not:`` in front of what must be false, a statement holds when the model's
probability is at the threshold, a picture's score is its weakest statement.

Nothing here is used without Ollama; a judge that cannot be asked is an error
the loop reports, never a verdict.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("agentY.condition")

MODELS = ("clef-flash", "clef")
MAX_STATEMENTS = 32
BATCH_ALL = "every image must pass"
BATCH_ANY = "any image may pass"
BATCH_FIRST = "first image only"
BATCH_MODES = (BATCH_ALL, BATCH_ANY, BATCH_FIRST)
VIDEO_FRAMES = 3
_VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


class ConditionError(RuntimeError):
    """The judge could not be asked. Says what to do about it."""


def parse(condition) -> list[tuple[str, bool]]:
    """``[(statement, must_be_true)]``, one per line of the node's text."""
    out = []
    for line in str(condition or "").splitlines():
        text = line.strip().lstrip("-*• ").strip()
        if not text or text.startswith("#"):
            continue
        want = True
        low = text.lower()
        for prefix in ("not:", "not :", "no:", "never:"):
            if low.startswith(prefix):
                want, text = False, text[len(prefix):].strip()
                break
        if text:
            out.append((text, want))
    return out[:MAX_STATEMENTS]


def settings(hook: dict | None) -> dict:
    """What a condition node on the wire says, as the panel reports it."""
    hook = hook or {}
    model = str(hook.get("model") or MODELS[0]).strip()
    try:
        threshold = float(hook.get("threshold"))
    except (TypeError, ValueError):
        threshold = 0.5
    batch = str(hook.get("batch") or BATCH_ALL).strip()
    return {
        "node_id": str(hook.get("hook_node_id") or ""),
        "condition": "\n".join(ln.strip() for ln in str(hook.get("condition") or "").splitlines() if ln.strip()),
        "model": model if model in MODELS else MODELS[0],
        "threshold": max(0.05, min(0.95, threshold)),
        "batch": batch if batch in BATCH_MODES else BATCH_ALL,
    }


def verdict(probabilities, statements, threshold: float) -> dict:
    """One picture's verdict from the model's probabilities."""
    rows = []
    for p, (text, want) in zip(probabilities, statements):
        holds = p if want else 1.0 - p
        rows.append({"statement": text, "wanted": want, "probability": round(p, 4),
                     "holds": round(holds, 4), "ok": holds >= threshold})
    score = min((r["holds"] for r in rows), default=0.0)
    return {"met": bool(rows) and all(r["ok"] for r in rows), "score": round(score, 4), "rows": rows}


def combine(verdicts: list, mode: str) -> dict:
    """Several pictures (the frames of a film) as one verdict."""
    if not verdicts:
        return {"met": False, "score": 0.0, "rows": []}
    if mode == BATCH_FIRST or len(verdicts) == 1:
        return verdicts[0]
    if mode == BATCH_ANY:
        return max(verdicts, key=lambda v: (v["met"], v["score"]))
    worst = min(verdicts, key=lambda v: (v["met"], v["score"]))
    return {**worst, "met": all(v["met"] for v in verdicts)}


def missed(v: dict) -> list[str]:
    """The statements a picture did not meet, as objections a next round can act on."""
    out = []
    for row in v.get("rows") or []:
        if row["ok"]:
            continue
        pct = int(round(100 * row["probability"]))
        out.append((f"{row['statement']}" if row["wanted"] else f"must NOT be the case: {row['statement']}")
                   + f" (the judge saw it at {pct} %)")
    return out


def ready(model: str, say=None, notify=None) -> None:
    """Have *model* there, downloading it on first use (with a progress bar).

    Raises :class:`ConditionError` when Ollama is not running or the download
    fails: a loop judged by a decision model cannot go on without it.
    """
    from src.utils import decision_models as dm
    if not dm.ollama_up():
        raise ConditionError(
            "the loop's condition node is judged by a decision model through Ollama, and Ollama "
            "is not running on this machine. Start Ollama, or take the condition node out of the "
            "loop and write the condition on the loop break (the QA agent judges that).")
    if dm.installed(model):
        return
    if not dm.settings().get("download", True):
        raise ConditionError(f"{model} is not installed and downloads are switched off "
                             f"(decision.download): `ollama pull {model}`.")
    if say is None:
        try:
            from agenty_core.utils.progress_signal import push as say
        except Exception:  # noqa: BLE001
            say = lambda _line: None  # noqa: E731
    if notify is None:
        try:
            from src.utils.status_bus import notify
        except Exception:  # noqa: BLE001
            notify = lambda _text, **_kw: None  # noqa: E731
    size = dm.SIZE_GB.get(model.split(":")[0])
    notify(f"⬇️ Downloading {model}" + (f" (about {size:g} GB)" if size else "")
           + " through Ollama for the loop's condition - first use.")
    if not dm._pull(model, "loop condition", dm._host(), say, notify):
        raise ConditionError(f"{model} could not be downloaded through Ollama: `ollama pull {model}`.")


def _pictures(path: str, mode: str) -> list[str]:
    if Path(path).suffix.lower() not in _VIDEO:
        return [path]
    from agenty_core.utils.video_frames import extract_frames
    frames = [str(f) for f in (extract_frames(Path(path), count=1 if mode == BATCH_FIRST else VIDEO_FRAMES) or [])]
    if not frames:
        raise ConditionError(f"no frame could be read from {Path(path).name}")
    return frames


def judge_file(path: str, judge: dict, ask=None, encode=None) -> dict:
    """*path* against the node's statements: ``{met, score, rows}``.

    *ask* and *encode* are ``decision_models.ask`` / ``_encoded``; passed in by tests.
    """
    statements = parse(judge.get("condition"))
    if not statements:
        raise ConditionError("the condition node in this loop has no statement written in it.")
    if ask is None or encode is None:
        from src.utils import decision_models as dm
        ask = ask or dm.ask
        encode = encode or dm._encoded
    questions = {f"s{i}": {"type": "noul", "instructions": f"Is this true of the picture: {text}"}
                 for i, (text, _want) in enumerate(statements)}
    verdicts = []
    for picture in _pictures(path, judge.get("batch") or BATCH_ALL):
        answers = ask(judge["model"], "Look at the attached picture and judge each statement about it.",
                      questions, images=[encode(picture)], timeout=600)
        probabilities = []
        for i in range(len(statements)):
            value = (answers.get(f"s{i}") or {}).get("noul")
            if not isinstance(value, (int, float)):
                raise ConditionError(f"{judge['model']} gave no answer to statement {i + 1}.")
            probabilities.append(float(value))
        verdicts.append(verdict(probabilities, statements, float(judge.get("threshold") or 0.5)))
    return combine(verdicts, judge.get("batch") or BATCH_ALL)
