"""Where an Anthropic request asks for its prompt to be cached.

Anthropic caches a request's prefix up to each ``cache_control`` breakpoint, in
the order tools -> system -> messages, and a later request reads it back only
when its bytes up to that point are the same. agentY's agents call the model in
a loop: each call is the last one plus the model's reply and the tool results.
Marking only the tools and the system prompt (as was done) cached ~60k tokens
and sent the whole growing conversation fresh on every call - measured over the
last fortnight of orchestrator calls (``.logs/tokens_usage.log``): 52% hit.

So a request gets at most four breakpoints:

1. the last tool                      - the tool list,
2. the last block of the STABLE system prompt - the prompt, its skills list;
   the turn's guidance (pipeline ``_set_turn_guides``, which changes from turn
   to turn) is moved behind it, so a new turn still reads the prompt,
3. the last block of the last message - written now, read by the next call,
4. the last block of the third-last message - where the previous call of the
   loop wrote its own (3), so it is read back exactly, however many blocks one
   round of tool calls added (the API only looks ~20 blocks back on its own).

The tools and the prompt live through the user's pauses between turns, so they
may take the one-hour TTL (``ttl_static``); the conversation's tail is written
on every call and keeps the five-minute one (a 1h write costs 2x, a 5m one 1.25x).
"""

from __future__ import annotations

from typing import Any

# The start of the pipeline's per-turn guidance block (src/pipeline.py
# _TURN_GUIDES_HEAD). Copied, not imported: the pipeline imports the agents.
TURN_GUIDES_HEAD = "\n\n# For this kind of turn\n\n"

MAX_BREAKPOINTS = 4

# Content blocks a breakpoint may sit on (not thinking blocks, not empty text).
_CACHEABLE = {"text", "image", "tool_use", "tool_result", "document"}


def cache_control(ttl: str | None = None) -> dict:
    """``{"type": "ephemeral"}``, with ``"ttl": "1h"`` for the one-hour cache."""
    return {"type": "ephemeral", "ttl": "1h"} if str(ttl or "").strip().lower() == "1h" else {"type": "ephemeral"}


def system_blocks(system_prompt: str, blocks: list | None = None, *, ttl: str | None = None) -> list[dict]:
    """The system prompt as Anthropic text blocks, the turn's guidance last and
    the breakpoint on the last block before it.

    *blocks* are Strands' system content blocks (``{"text": ...}``); without them
    the prompt is one block."""
    texts = [str(b["text"]) for b in (blocks or []) if isinstance(b, dict) and b.get("text")]
    if not texts:
        texts = [system_prompt] if system_prompt else []
    stable = [t for t in texts if not t.startswith(TURN_GUIDES_HEAD)]
    volatile = [t for t in texts if t.startswith(TURN_GUIDES_HEAD)]
    out = [{"type": "text", "text": t} for t in stable + volatile]
    mark = len(stable) - 1 if stable else len(out) - 1
    if mark >= 0:
        out[mark]["cache_control"] = cache_control(ttl)
    return out


def _count(blocks: Any) -> int:
    return sum(1 for b in blocks or [] if isinstance(b, dict) and b.get("cache_control"))


def _mark_message(message: dict) -> bool:
    """Put a breakpoint on the last block of *message* that can take one."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    if any(isinstance(b, dict) and b.get("cache_control") for b in content):
        return False                        # already one there (a Strands cachePoint)
    for i in range(len(content) - 1, -1, -1):
        b = content[i]
        if not isinstance(b, dict) or b.get("type") not in _CACHEABLE:
            continue
        if b.get("type") == "text" and not str(b.get("text") or "").strip():
            continue
        content[i] = {**b, "cache_control": cache_control()}
        return True
    return False


def apply(req: dict, system_prompt: str | None = None, system_content: list | None = None,
          *, ttl_static: str | None = None) -> dict:
    """Mark *req* (an Anthropic messages request) for caching, as described above.

    *system_prompt* / *system_content* are the prompt of the call; when neither
    has text, the request's own ``system`` stays as it is. Never more than
    :data:`MAX_BREAKPOINTS`: a request with more is refused by the API."""
    if (isinstance(system_prompt, str) and system_prompt.strip()) or system_content:
        req["system"] = system_blocks(system_prompt or "", system_content, ttl=ttl_static)
    tools = req.get("tools")
    if tools:
        *head, last = tools
        req["tools"] = head + [{**last, "cache_control": cache_control(ttl_static)}]

    system = req.get("system")
    used = _count(req.get("tools")) + (_count(system) if isinstance(system, list) else 0)
    messages = req.get("messages") or []
    used += sum(_count(m.get("content")) for m in messages if isinstance(m, dict))
    for idx in (-1, -3):
        if used >= MAX_BREAKPOINTS or len(messages) < -idx:
            break
        if isinstance(messages[idx], dict) and _mark_message(messages[idx]):
            used += 1
    return req
