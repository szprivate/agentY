"""
agentY – Self-learning layer: analyses Brain message history after long sessions.

After any Brain session where the agent used **more than 5 tool calls**, this
module fires the ``learnings agent`` asynchronously.  The learnings agent
scans the message history for repeated failure→fix patterns and appends concise
1–2-sentence learnings to ``skills/brain-learnings/SKILL.md``.

The agent also leverages the FAISS memory layer to cross-reference previously
stored learnings, avoiding duplicate entries across sessions.

Public API
----------
>>> from src.utils.learnings import count_tool_calls, maybe_run_learnings
>>> n = count_tool_calls(brain_agent.messages)
>>> if n > 5:
...     maybe_run_learnings(brain_agent.messages, session_id=session_id)
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any

# Path to the self-learning skill file that the learnings agent appends to.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SKILL_PATH = _PROJECT_ROOT / "skills" / "brain-learnings" / "SKILL.md"

# Hard cap on conversation text sent to the learnings agent to avoid blowing up
# the context window of a small local model.
_MAX_HISTORY_CHARS = 20_000


# ---------------------------------------------------------------------------
# Tool-call counter
# ---------------------------------------------------------------------------

def count_tool_calls(messages: list[dict]) -> int:
    """Return the number of tool-use blocks in *messages*.

    Handles both the Strands/Bedrock Converse format (``{"toolUse": {...}}``) and
    the Anthropic native format (``{"type": "tool_use", ...}``), plus a plain-text
    fallback for serialised message strings.
    """
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                # Strands/Bedrock Converse format
                if "toolUse" in item:
                    count += 1
                # Anthropic native format
                elif item.get("type") == "tool_use":
                    count += 1
        # Plain-text fallback: check if a serialised tool_use marker exists
        elif isinstance(content, str):
            count += content.count('"toolUse"')
            count += content.count('"type": "tool_use"')
    return count


# ---------------------------------------------------------------------------
# Message formatter — converts Strands/Anthropic message list → readable text
# ---------------------------------------------------------------------------

def _text_from_content(content: Any) -> str:
    """Extract a plain-text representation from a message ``content`` value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            t = item.get("type", "")
            # ── Strands/Bedrock Converse format ───────────────────────────
            if "toolUse" in item:
                tu = item["toolUse"]
                name = tu.get("name", "?")
                inp = tu.get("input", {})
                try:
                    inp_str = json.dumps(inp, ensure_ascii=False)[:600]
                except Exception:
                    inp_str = str(inp)[:600]
                parts.append(f"[TOOL CALL → {name}] {inp_str}")
            elif "toolResult" in item:
                tr = item["toolResult"]
                status = tr.get("status", "")
                rc = tr.get("content", "")
                if isinstance(rc, list):
                    result_str = " ".join(
                        c.get("text", str(c)) if isinstance(c, dict) else str(c)
                        for c in rc
                    )[:800]
                else:
                    result_str = str(rc)[:800]
                parts.append(f"[TOOL RESULT:{status}] {result_str}")
            # ── Anthropic native format ───────────────────────────────────
            elif t == "text":
                parts.append(item.get("text", ""))
            elif t == "tool_use":
                name = item.get("name", "?")
                inp = item.get("input", {})
                try:
                    inp_str = json.dumps(inp, ensure_ascii=False)[:600]
                except Exception:
                    inp_str = str(inp)[:600]
                parts.append(f"[TOOL CALL → {name}] {inp_str}")
            elif t == "tool_result":
                result_content = item.get("content", "")
                if isinstance(result_content, list):
                    result_parts = [
                        c.get("text", str(c)) if isinstance(c, dict) else str(c)
                        for c in result_content
                    ]
                    result_str = " ".join(result_parts)[:800]
                else:
                    result_str = str(result_content)[:800]
                parts.append(f"[TOOL RESULT] {result_str}")
            elif not t:
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p.strip())
    return str(content)


def format_messages_as_transcript(messages: list[dict]) -> str:
    """Render *messages* as a human-readable transcript string.

    Each message is prefixed with its role.  The output is capped at
    ``_MAX_HISTORY_CHARS`` to protect small LLM context windows.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        text = _text_from_content(msg.get("content", ""))
        if text.strip():
            lines.append(f"[{role}]\n{text}")
    transcript = "\n\n---\n\n".join(lines)
    if len(transcript) > _MAX_HISTORY_CHARS:
        transcript = transcript[:_MAX_HISTORY_CHARS] + "\n\n[... transcript truncated ...]"
    return transcript


# ---------------------------------------------------------------------------
# SKILL.md writer
# ---------------------------------------------------------------------------

def _append_to_skill(entries: str) -> None:
    """Append *entries* (one line per learning) to the SKILL.md learnings log.

    Entries that already appear verbatim in the file are silently skipped to
    prevent duplicates caused by re-runs.
    """
    _SKILL_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _SKILL_PATH.read_text(encoding="utf-8") if _SKILL_PATH.exists() else ""

    new_lines: list[str] = []
    for line in entries.splitlines():
        line = line.strip()
        if not line or line == "NO_NEW_LEARNINGS":
            continue
        # Skip if an identical entry already exists (basic dedup).
        # Compare only the problem|solution portion (after the date prefix).
        parts = line.split("|", 1)
        key = parts[1].strip() if len(parts) > 1 else line
        if key in existing:
            continue
        new_lines.append(line)

    if not new_lines:
        return

    append_block = "\n" + "\n".join(new_lines) + "\n"
    with _SKILL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(append_block)
    print(f"[learnings] Appended {len(new_lines)} new learning(s) to {_SKILL_PATH}")


# ---------------------------------------------------------------------------
# Core async execution
# ---------------------------------------------------------------------------

async def _run_learnings_async(messages: list[dict], session_id: str) -> None:
    """Analyse *messages* and append any new learnings to the SKILL.md file.

    This coroutine is fire-and-forget — it should never raise to the caller.
    """
    try:
        tool_count = count_tool_calls(messages)
        print(f"[learnings] Brain used {tool_count} tool calls — analysing for learnings …")

        transcript = format_messages_as_transcript(messages)
        today = datetime.date.today().isoformat()

        # Retrieve relevant past learnings from FAISS memory for deduplication context.
        past_learnings_block = ""
        try:
            from src.utils.memory import memory_search, memory_add
            past_hits = memory_search(
                "brain tool call failure pattern solution",
                session_id="learnings_global",
                limit=10,
            )
            if past_hits:
                past_texts = [h.get("memory") or h.get("text", "") for h in past_hits]
                past_learnings_block = (
                    "\n\n## Past learnings from long-term memory (dedup reference)\n\n"
                    + "\n".join(f"- {t}" for t in past_texts if t)
                )
        except Exception as mem_exc:
            print(f"[learnings] Warning: could not retrieve past learnings from memory: {mem_exc}")

        # Build the prompt for the learnings agent.
        prompt = (
            f"Today's date: {today}\n\n"
            f"## Brain session transcript ({tool_count} tool calls)\n\n"
            f"{transcript}"
            f"{past_learnings_block}\n\n"
            "Analyse the transcript above and output any new learnings in the required format."
        )

        # Lazily import to avoid circular-import issues.
        from src.agent import create_learnings_agent
        agent = create_learnings_agent()
        response = str(agent(prompt)).strip()

        if not response or response == "NO_NEW_LEARNINGS":
            print("[learnings] No new learnings found for this session.")
            return

        # Persist new learnings to the SKILL.md file.
        _append_to_skill(response)

        # Also store each learning as a FAISS memory for cross-session recall.
        try:
            from src.utils.memory import memory_add
            for line in response.splitlines():
                line = line.strip()
                if line and line != "NO_NEW_LEARNINGS":
                    memory_add(line, session_id="learnings_global", metadata={"source": "learnings_agent"})
        except Exception as mem_exc:
            print(f"[learnings] Warning: could not store learnings in FAISS memory: {mem_exc}")

    except Exception as exc:
        # Never raise — this is a background best-effort enhancement.
        print(f"[learnings] WARNING: learnings agent failed: {exc}")


def _run_learnings_in_thread(messages: list[dict], session_id: str) -> None:
    """Run learnings analysis in a background daemon thread (fire-and-forget)."""
    import asyncio

    def _target() -> None:
        try:
            asyncio.run(_run_learnings_async(messages, session_id))
        except Exception as exc:
            print(f"[learnings] Background thread error: {exc}")

    thread = threading.Thread(target=_target, daemon=True, name="learnings-agent")
    thread.start()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def maybe_run_learnings(
    messages: list[dict],
    session_id: str = "default",
    threshold: int = 5,
) -> bool:
    """Start the learnings agent in a background thread if tool-call count > threshold.

    Args:
        messages:   Brain agent message history (Strands/Anthropic format).
        session_id: Pipeline session ID (used for FAISS memory namespace).
        threshold:  Minimum tool-call count to trigger analysis (default: 5).

    Returns:
        ``True`` if the learnings agent was started, ``False`` otherwise.
    """
    tool_count = count_tool_calls(messages)
    if tool_count <= threshold:
        if tool_count > 0:
            print(f"[learnings] {tool_count} tool call(s) — below threshold ({threshold}); skipping.")
        return False

    # Snapshot the messages so the background thread is not affected by
    # the caller mutating the list during history compression.
    messages_snapshot = list(messages)
    _run_learnings_in_thread(messages_snapshot, session_id)
    print(f"[learnings] {tool_count} tool calls — learnings agent started in background.")
    return True


def record_user_advice_learning(
    error_context: str,
    user_advice: str,
    session_id: str = "default",
) -> None:
    """Record a learning derived from a Brain failure that was resolved by user advice.

    Fires in a background thread (fire-and-forget).  Calls the learnings agent
    to produce a concise entry from the error and advice, then appends it to
    ``skills/brain-learnings/SKILL.md`` and the FAISS memory store.

    Args:
        error_context: Short description of what the Brain failed to do.
        user_advice:   The advice the user provided that led to success.
        session_id:    Pipeline session ID (used for FAISS memory namespace).
    """
    import asyncio

    async def _run() -> None:
        try:
            today = datetime.date.today().isoformat()
            prompt = (
                f"Today's date: {today}\n\n"
                f"## Brain assembly failure resolved by user advice\n\n"
                f"**Original error:** {error_context}\n\n"
                f"**User's advice (that led to success):** {user_advice}\n\n"
                f"Produce exactly one learning entry in this format:\n"
                f"YYYY-MM-DD | <problem: ≤15 words> | <solution: 1–2 sentences, ≤40 words>\n"
                f"Use today's date. Plain text only — no markdown, no preamble."
            )
            from src.agent import create_learnings_agent
            agent = create_learnings_agent()
            response = str(agent(prompt)).strip()
            if not response or response == "NO_NEW_LEARNINGS":
                return
            _append_to_skill(response)
            try:
                from src.utils.memory import memory_add
                for line in response.splitlines():
                    line = line.strip()
                    if line and line != "NO_NEW_LEARNINGS":
                        memory_add(line, session_id="learnings_global", metadata={"source": "learnings_agent"})
            except Exception as mem_exc:
                print(f"[learnings] Warning: could not store user-advice learning in FAISS: {mem_exc}")
        except Exception as exc:
            print(f"[learnings] WARNING: record_user_advice_learning failed: {exc}")

    def _target() -> None:
        try:
            asyncio.run(_run())
        except Exception as exc:
            print(f"[learnings] Background thread error (user-advice learning): {exc}")

    thread = threading.Thread(target=_target, daemon=True, name="learnings-user-advice")
    thread.start()
    print("[learnings] User-advice learning agent started in background.")
