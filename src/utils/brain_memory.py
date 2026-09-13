"""A conversation's working memory, kept across a restart of the host.

The orchestrator's message list IS the conversation as the agent knows it: what
was asked, what it tried, what its tools said. It lived in exactly one place — a
dict in this process — so every restart of the host wiped it, and the next
message in an old conversation reached an agent with no idea what had been going
on. The transcript was on disk the whole time; nothing ever read it back.

Two things here close that:

* :func:`serialize_messages` turns the live message list into something that
  survives ``json.dumps`` — images and other binary blocks become a line saying
  they were there, reasoning blocks are dropped (a signature minted by one model
  is rejected by the next), and oversized tool output is clipped so the saved
  state stays small.
* :func:`transcript_to_messages` rebuilds a history from the visible transcript,
  for conversations saved before there was any saved state. Only the words
  survive that route, and the first message says so.

:func:`choose_history` is the order they are tried in.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# Tool output beyond this is clipped when saved. What a tool said three turns ago
# is worth keeping as context; the whole of a template dump is not, and the state
# is rewritten after every turn.
MAX_TOOL_TEXT = 6000

_BINARY_NOTE = {
    "image": "(an image was attached here — not kept in saved history)",
    "document": "(a document was attached here — not kept in saved history)",
    "video": "(a video was attached here — not kept in saved history)",
}

RESTORED_NOTE = (
    "[Earlier in this conversation, restored from the saved transcript after the "
    "agent restarted. Only the messages survived — the tool steps behind them "
    "did not.]\n\n"
)


def _clip(text: Any, limit: int = MAX_TOOL_TEXT) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters not kept]"


def _jsonable(value: Any) -> Any:
    """*value* with anything ``json.dumps`` would choke on made printable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "(binary data not kept)"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _tool_result_content(items: Any) -> list:
    out: list = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            out.append({"text": _clip(item["text"])})
        elif "json" in item:
            payload = _jsonable(item["json"])
            dumped = json.dumps(payload)
            out.append({"text": _clip(dumped)} if len(dumped) > MAX_TOOL_TEXT
                       else {"json": payload})
        else:
            kind = next((k for k in _BINARY_NOTE if k in item), None)
            if kind:
                out.append({"text": _BINARY_NOTE[kind]})
    return out or [{"text": "(no output)"}]


def _block(block: Any) -> dict | None:
    if not isinstance(block, dict):
        return None
    if "text" in block:
        return {"text": str(block["text"])}
    if "toolUse" in block:
        tu = block["toolUse"] if isinstance(block["toolUse"], dict) else {}
        return {"toolUse": {"toolUseId": str(tu.get("toolUseId", "")),
                            "name": str(tu.get("name", "")),
                            "input": _jsonable(tu.get("input", {}))}}
    if "toolResult" in block:
        tr = block["toolResult"] if isinstance(block["toolResult"], dict) else {}
        result = {"toolUseId": str(tr.get("toolUseId", "")),
                  "content": _tool_result_content(tr.get("content"))}
        if tr.get("status"):
            result["status"] = str(tr["status"])
        return {"toolResult": result}
    for kind, note in _BINARY_NOTE.items():
        if kind in block:
            return {"text": note}
    # reasoningContent (its signature is valid only for the model that wrote it),
    # cachePoint, and anything unrecognised: nothing a resumed conversation needs.
    return None


def serialize_messages(messages: Iterable[Any] | None) -> list[dict]:
    """A JSON-safe copy of an agent's messages, fit to be saved and loaded back.

    Every message keeps its place, so a tool call and its result stay paired: a
    message whose every block was dropped becomes ``(empty)`` rather than
    vanishing and taking the conversation's turn order with it.
    """
    out: list[dict] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            blocks = [{"text": content}]
        elif isinstance(content, list):
            blocks = [b for b in (_block(x) for x in content) if b is not None]
        else:
            continue
        out.append({"role": msg["role"], "content": blocks or [{"text": "(empty)"}]})
    return out


def transcript_to_messages(rows: Iterable[dict] | None, *, max_messages: int = 20,
                           max_chars: int = 24000) -> list[dict]:
    """A history rebuilt from the saved transcript (oldest row first).

    The transcript already holds the message that started THIS turn — the chat
    route saves it before the turn runs — so anything after the last reply is left
    out: otherwise the agent would read the new message twice, once as history and
    once as the request. Slash commands are not conversation and are skipped.
    Newest exchanges win the budget.
    """
    turns: list[dict] = []
    for row in rows or []:
        role = (row or {}).get("role")
        text = str((row or {}).get("content") or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if role == "user" and text.startswith("/"):
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["text"] += "\n\n" + text
        else:
            turns.append({"role": role, "text": text})
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()

    kept: list[dict] = []
    total = 0
    for turn in reversed(turns):
        text = _clip(turn["text"], max_chars)
        if len(kept) >= max_messages or (kept and total + len(text) > max_chars):
            break
        kept.append({"role": turn["role"], "text": text})
        total += len(text)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    if not kept:
        return []
    kept[0]["text"] = RESTORED_NOTE + kept[0]["text"]
    return [{"role": t["role"], "content": [{"text": t["text"]}]} for t in kept]


def choose_history(cached: list | None, saved: Any,
                   transcript_rows: Iterable[dict] | None) -> tuple[list, str]:
    """``(messages, where_they_came_from)`` for a conversation being resumed.

    In order: this process's own copy (exact); the saved state (survives a
    restart); the transcript (conversations saved before there was saved state).
    ``"none"`` means there is no history — a new conversation.
    """
    if cached is not None:
        return list(cached), "cache"
    if isinstance(saved, list) and saved:
        return [m for m in saved if isinstance(m, dict)], "saved"
    rebuilt = transcript_to_messages(transcript_rows)
    return rebuilt, ("transcript" if rebuilt else "none")
