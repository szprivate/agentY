"""Keep a conversation's working history small without losing the conversation.

A conversation used to be bounded by one rule: the orchestrator kept its last 12
messages, and everything older was dropped — silently, with no summary, so the
agent forgot the start of long conversations. Inside a turn nothing was bounded
at all. Measured on a real conversation (2026-10-05), history was 80% old tool
output: dumps the agent had read once and would never read again.

Two steps, cheapest first, both leaving a valid message list behind:

1. **Age.** Tool output that is no longer recent is cut to its head, with a line
   saying it was cut. Injected per-turn blocks in old user messages (the canvas
   guidance for a turn long over) get the same. No model call.
2. **Summarise.** If history is still over budget, everything before the last few
   turns becomes one "earlier in this conversation" block at the top of the first
   message kept. The cut is always at a real user message, so no tool call is
   ever separated from its result.

**Nothing is destroyed.** Every message a step changes or removes is first
appended, whole, to ``memory/history_archive/<thread>.jsonl``.

**A restart resumes from the same place.** What is compacted is the list that is
saved as the conversation's state (see ``brain_memory``), so after a crash the
agent comes back with exactly the context it had.

This module is pure: lists in, lists out. When it runs — between turns in the
background, and as a guard mid-turn — is the caller's business
(``agentY_server`` and :class:`CompactionHookProvider`).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("agentY.compaction")

_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_DIR = _ROOT / "memory" / "history_archive"

NEWLINE = chr(10)
TRIM_MARK = "[trimmed by compaction:"
SUMMARY_MARK = "[EARLIER IN THIS CONVERSATION — summarised to keep it short"
SUMMARY_END = "[END OF SUMMARY — the conversation continues below]"

DEFAULTS = {
    "enabled": True,
    # History (messages only — the system prompt and tools are not counted) the
    # between-turns pass aims to stay under before it summarises older turns.
    "history_budget_tokens": 20_000,
    # Mid-turn guard: past this, old tool output is aged even while a turn runs.
    "hard_budget_tokens": 60_000,
    "keep_recent_messages": 8,     # tool output this recent is never aged
    "keep_turns": 2,               # user turns kept word for word by a summary
    "tool_result_chars": 1_500,    # tool output longer than this can be aged …
    "tool_result_head": 500,       # … down to this much of its start
    # An old turn's input is one block: injected guidance for that turn, then
    # the user's own words LAST. Past this length it is cut to its start and its
    # end, so what the user actually wrote is always kept.
    "turn_input_chars": 4_000,
    "turn_input_head": 400,
    "turn_input_tail": 2_500,
}


def settings() -> dict:
    """``[compaction]`` from settings over the defaults."""
    out = dict(DEFAULTS)
    try:
        from src.utils.settings import load_settings
        for key, value in (load_settings().get("compaction") or {}).items():
            if key in out:
                out[key] = type(DEFAULTS[key])(value)
    except Exception as exc:  # noqa: BLE001 — settings never break a turn
        logger.debug("compaction: settings unreadable (%s)", exc)
    return out


# ── measuring ────────────────────────────────────────────────────────────────

def estimate_tokens(messages) -> int:
    """Roughly how many tokens *messages* cost: serialised characters / 4."""
    total = 0
    for m in messages or []:
        try:
            total += len(json.dumps(m, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            total += len(str(m))
    return total // 4


def _is_turn_start(message) -> bool:
    """A message the USER wrote: user role, has text, carries no tool result."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content") or []
    if any(isinstance(c, dict) and "toolResult" in c for c in content):
        return False
    return any(isinstance(c, dict) and str(c.get("text") or "").strip() for c in content)


def turn_starts(messages) -> list[int]:
    return [i for i, m in enumerate(messages or []) if _is_turn_start(m)]


# ── step 1: age ──────────────────────────────────────────────────────────────

def _cut(text: str, head: int, what: str) -> str:
    return (f"{TRIM_MARK} {len(text) - head:,} characters of {what} removed; the start "
            f"is kept below. Run the tool again if you need the rest.]\n" + text[:head])


def _cut_input(text: str, head: int, tail: int) -> str:
    """An old turn's input, without the guidance that applied to that turn only."""
    return (f"{TRIM_MARK} {len(text) - head - tail:,} characters of context injected for "
            f"this earlier turn removed — it applied to that turn only. Its start, and the "
            f"user's own message at the end, are kept.]\n"
            + text[:head] + "\n[…]\n" + text[-tail:])


def age(messages: list, *, keep_recent: int, tool_chars: int, tool_head: int,
        input_chars: int, input_head: int, input_tail: int) -> tuple[list, list]:
    """Cut old tool output (and old injected turn blocks) to their heads.

    Returns ``(new_messages, originals_changed)``. Messages are copied only where
    they change; a message already aged is left alone, so this is idempotent.
    The last *keep_recent* messages are never touched, and neither is anything
    from the newest user turn on (its injected blocks are still in force).
    """
    out = list(messages or [])
    changed: list = []
    starts = turn_starts(out)
    live_from = starts[-1] if starts else len(out)
    limit = min(len(out) - max(0, keep_recent), len(out))
    for i in range(max(0, limit)):
        m = out[i]
        if not isinstance(m, dict):
            continue
        new_content, touched = [], False
        for block in m.get("content") or []:
            nb = block
            if isinstance(block, dict) and "toolResult" in block:
                tr = block["toolResult"]
                items, hit = [], False
                for item in tr.get("content") or []:
                    text = None
                    if isinstance(item, dict) and "text" in item:
                        text = str(item["text"])
                    elif isinstance(item, dict) and "json" in item:
                        text = json.dumps(item["json"], ensure_ascii=False, default=str)
                    if text is not None and len(text) > tool_chars \
                            and not text.startswith(TRIM_MARK):
                        items.append({"text": _cut(text, tool_head, "tool output")})
                        hit = True
                    else:
                        items.append(item)
                if hit:
                    nb = {"toolResult": {**tr, "content": items}}
                    touched = True
            elif (isinstance(block, dict) and "text" in block and m.get("role") == "user"
                  and i < live_from):
                text = str(block["text"])
                if len(text) > input_chars and not text.startswith((TRIM_MARK, SUMMARY_MARK)):
                    nb = {**block, "text": _cut_input(text, input_head, input_tail)}
                    touched = True
            new_content.append(nb)
        if touched:
            changed.append(m)
            out[i] = {**m, "content": new_content}
    return out, changed


# ── step 2: summarise ────────────────────────────────────────────────────────

def _clip(text, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def render_for_summary(messages, limit: int = 40_000) -> str:
    """The messages to be replaced, as a transcript a small model can read."""
    lines: list[str] = []
    for m in messages or []:
        role = str(m.get("role") or "")
        for block in m.get("content") or []:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text = str(block["text"])
                if text.startswith(SUMMARY_MARK):
                    body = text[len(SUMMARY_MARK):].split(SUMMARY_END)[0]
                    lines.append("PREVIOUS SUMMARY:\n" + body.strip(" ]\n:"))
                    rest = text.split(SUMMARY_END, 1)[-1].strip() if SUMMARY_END in text else ""
                    if rest:
                        lines.append(f"{role.upper()}: {_clip(rest, 1500)}")
                else:
                    lines.append(f"{role.upper()}: {_clip(text, 1500)}")
            elif "toolUse" in block:
                tu = block["toolUse"]
                lines.append(f"  -> {tu.get('name')}({_clip(json.dumps(tu.get('input'), default=str), 200)})")
            elif "toolResult" in block:
                parts = [str(i.get("text") or i.get("json") or "")
                         for i in block["toolResult"].get("content") or [] if isinstance(i, dict)]
                lines.append(f"  <- {_clip(' '.join(parts), 300)}")
    text = "\n".join(lines)
    if len(text) > limit:
        half = limit // 2
        text = text[:half] + "\n… (middle left out) …\n" + text[-half:]
    return text


def digest(messages) -> str:
    """A summary with no model: what was asked and how each turn ended.

    The fallback when the summarising model cannot be reached — worse than a
    written summary, far better than dropping the turns with nothing.
    """
    lines = ["(written without a model — the summariser was unavailable)"]
    starts = turn_starts(messages)
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(messages)
        # The user's own words are the END of a turn's input (injected guidance
        # comes first, and an aged block keeps its tail), so read from the end.
        asked = " ".join(str(c.get("text") or "") for c in messages[start].get("content") or []
                         if isinstance(c, dict))
        asked = asked.rstrip().rsplit(NEWLINE * 2, 1)[-1]
        answer = ""
        for m in reversed(messages[start:end]):
            if m.get("role") == "assistant":
                answer = " ".join(str(c.get("text") or "") for c in m.get("content") or []
                                  if isinstance(c, dict) and "text" in c)
                if answer.strip():
                    break
        lines.append(f"- User: {_clip(asked, 400)}")
        if answer.strip():
            lines.append(f"  Agent: {_clip(answer, 400)}")
    return "\n".join(lines)


def llm_summary(messages) -> str:
    """The summary, written by the cheap utility model. Raises if it cannot."""
    import asyncio
    from src.utils.llm_functions import LLMFunctions
    system = (_ROOT / "config" / "system_prompts" / "system_prompt.compaction.md").read_text(
        encoding="utf-8")
    chat = [{"role": "system", "content": system},
            {"role": "user", "content": render_for_summary(messages)}]
    text = asyncio.run(LLMFunctions.from_settings().chat(chat))
    text = str(text or "").strip()
    if len(text) < 40:
        raise RuntimeError("the summariser returned nothing usable")
    return text


def summarise_older(messages: list, *, keep_turns: int,
                    summarise: Callable[[list], str] | None = None) -> tuple[list, list]:
    """Replace everything before the last *keep_turns* user turns with a summary.

    Returns ``(new_messages, originals_removed)``; unchanged when there are not
    more than *keep_turns* turns. The summary goes in as the first text block of
    the first message kept — a real user message — so roles still alternate and
    no tool call loses its result.
    """
    starts = turn_starts(messages)
    if len(starts) <= max(1, keep_turns):
        return list(messages), []
    cut = starts[-max(1, keep_turns)]
    old, recent = list(messages[:cut]), list(messages[cut:])
    if not old:
        return list(messages), []
    try:
        text = (summarise or llm_summary)(old)
    except Exception as exc:  # noqa: BLE001 — a summary must never cost the turns
        logger.warning("compaction: summariser failed (%s) — using the digest", exc)
        text = digest(old)
    first = copy.copy(recent[0])
    block = {"text": f"{SUMMARY_MARK}; the full messages are archived on disk]\n"
                     f"{text.strip()}\n{SUMMARY_END}"}
    first["content"] = [block] + list(first.get("content") or [])
    return [first] + recent[1:], old


# ── both steps ───────────────────────────────────────────────────────────────

@dataclass
class Result:
    messages: list
    before: int = 0
    after: int = 0
    aged: int = 0
    summarised: int = 0
    archived: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.aged or self.summarised)

    def describe(self) -> str:
        bits = []
        if self.aged:
            bits.append(f"{self.aged} old tool result(s)/input(s) trimmed")
        if self.summarised:
            bits.append(f"{self.summarised} older message(s) summarised")
        return (f"{', '.join(bits)}: ~{self.before:,} → ~{self.after:,} tokens"
                if bits else "nothing to compact")


def compact(messages: list, cfg: dict | None = None, *, allow_summary: bool = True,
            summarise: Callable[[list], str] | None = None) -> Result:
    """Age, then summarise if still over the history budget.

    *allow_summary* is False for the mid-turn guard: a model call in the middle
    of someone's turn is a wait they did not ask for, and cutting the history at
    a turn boundary while the turn is running would drop what it is working on.
    """
    cfg = cfg or settings()
    before = estimate_tokens(messages)
    aged, changed = age(messages, keep_recent=cfg["keep_recent_messages"],
                        tool_chars=cfg["tool_result_chars"], tool_head=cfg["tool_result_head"],
                        input_chars=cfg["turn_input_chars"], input_head=cfg["turn_input_head"],
                        input_tail=cfg["turn_input_tail"])
    res = Result(messages=aged, before=before, aged=len(changed), archived=list(changed))
    if allow_summary and estimate_tokens(aged) > cfg["history_budget_tokens"]:
        summed, removed = summarise_older(aged, keep_turns=cfg["keep_turns"],
                                          summarise=summarise)
        if removed:
            # Archive the messages as they were BEFORE ageing, not their stubs.
            by_id = {id(new): old for new, old in zip(aged, messages) if new is not old}
            res.archived = [m for m in res.archived
                            if all(m is not by_id.get(id(r)) for r in removed)]
            res.archived += [by_id.get(id(r), r) for r in removed]
            res.messages, res.summarised = summed, len(removed)
    res.after = estimate_tokens(res.messages)
    return res


# ── the archive ──────────────────────────────────────────────────────────────

def archive(thread_id: str, messages: list, reason: str = "") -> Path | None:
    """Append *messages*, whole, to this conversation's archive. Never raises."""
    if not messages or not thread_id:
        return None
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in str(thread_id) if c.isalnum() or c in "-_")[:80]
        path = ARCHIVE_DIR / f"{safe}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason,
                                 "messages": messages}, ensure_ascii=False,
                                default=lambda o: f"({type(o).__name__} not kept)") + "\n")
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("compaction: could not archive for %s (%s)", thread_id, exc)
        return None


# ── the mid-turn guard ───────────────────────────────────────────────────────

class CompactionHookProvider:
    """Age old tool output when one turn's history passes the hard budget.

    Between turns the server compacts in the background. This is for the turn
    that does not end: forty tool calls deep, each result still in full. It only
    ever AGES (no model call, no cut at a turn boundary), and only past the hard
    budget — so in an ordinary turn the history is byte-stable and the
    provider's prompt cache keeps hitting.
    """

    def __init__(self, role: str = "agent") -> None:
        self._role = role

    def register_hooks(self, registry, **kwargs) -> None:  # noqa: ARG002
        from strands.hooks.events import BeforeModelCallEvent
        registry.add_callback(BeforeModelCallEvent, self._before_model)

    def _before_model(self, event, **kwargs) -> None:  # noqa: ARG002
        try:
            cfg = settings()
            if not cfg["enabled"]:
                return
            agent = getattr(event, "agent", None)
            messages = getattr(agent, "messages", None)
            if not messages or estimate_tokens(messages) <= cfg["hard_budget_tokens"]:
                return
            res = compact(list(messages), cfg, allow_summary=False)
            if not res.changed:
                return
            if self._role == "orchestrator":
                archive(_current_thread(), res.archived, "mid-turn")
            messages[:] = res.messages
            logger.info("compaction (%s, mid-turn): %s", self._role, res.describe())
        except Exception as exc:  # noqa: BLE001 — never cost the model call
            logger.debug("compaction hook skipped (%s)", exc)


def _current_thread() -> str:
    try:
        from src.utils import chat_summary
        from agenty_core.utils import turn_scope
        scope = turn_scope.current()
        if scope is not turn_scope.DEFAULT:
            got = scope.get("chat_summary.thread")
            if got:
                return str(got)
        return str(getattr(chat_summary, "_current_thread", "") or "")
    except Exception:  # noqa: BLE001
        return ""
