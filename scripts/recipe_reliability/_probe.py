"""Minimal probe: confirm the pipeline can be driven headless (no Chainlit,
no console input()) by consuming stream_async and auto-answering its prompts.

Run:  python -m scripts.recipe_reliability._probe "generate an image of a red cube"
"""
from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
load_dotenv(os.path.join(_root, ".env"))

from src.pipeline import create_pipeline  # noqa: E402


async def drive(pipeline, user_input: str, timeout: float = 360.0):
    """Run one request headless. Returns (events, response_text)."""
    qa_q: asyncio.Queue = asyncio.Queue()
    seen: list[str] = []
    parts: list[str] = []
    in_researcher = False

    async def _consume():
        nonlocal in_researcher
        async for event in pipeline.stream_async(user_input, qa_reply_queue=qa_q):
            if not isinstance(event, dict):
                continue
            if event.get("brain_assembly_fail_ask"):
                seen.append("brain_assembly_fail_ask")
                await qa_q.put("")          # abort -> records a build failure
                continue
            if event.get("qa_fail_ask"):
                seen.append("qa_fail_ask")
                await qa_q.put("n")         # do not retry
                continue
            if event.get("approval_ask"):
                seen.append("approval_ask")
                await qa_q.put("y")         # approve / continue
                continue
            if event.get("_researcher_start"):
                in_researcher = True
                continue
            if event.get("_researcher_done"):
                in_researcher = False
                continue
            data = event.get("data")
            if data and not in_researcher:
                parts.append(data)
        await pipeline._await_pending_compression()

    await asyncio.wait_for(_consume(), timeout=timeout)
    return seen, "".join(parts)


def main() -> int:
    intent = sys.argv[1] if len(sys.argv) > 1 else "generate an image of a red cube"
    print(f"[probe] intent: {intent!r}")
    pipeline = create_pipeline(session_id="recipe-probe", verbose=False)
    try:
        seen, response = asyncio.run(drive(pipeline, intent))
    except asyncio.TimeoutError:
        print("[probe] TIMEOUT - driver did not complete in time")
        return 1
    print(f"[probe] interactive events seen: {seen or '(none)'}")
    print(f"[probe] response ({len(response)} chars):")
    print(response[-1500:])
    # Peek at the brain's last tool activity to see how outcome would be classified.
    try:
        msgs = pipeline._brain.messages
        print(f"[probe] brain message count: {len(msgs)}")
    except Exception as e:  # noqa: BLE001
        print(f"[probe] (could not read brain messages: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
