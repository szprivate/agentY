"""Wake the agent for a message that arrives while ComfyUI is rendering.

After the orchestrator signals its workflows, the turn spends most of its time in
the executor: queue waits, sampler steps, healing. The orchestrator is idle for
all of it — its loop has ended — so a message sent then had no tool boundary to
ride on and waited for the whole render, which is most of a turn.

:func:`interleave_checkins` runs the executor's progress stream as before and, in
the gaps while it waits for the next line, checks whether the user has said
something. If so it runs a check-in — the orchestrator reading the message and
acting on it — and carries on with the render, which never stopped: ComfyUI and
the executor's monitors keep going while the check-in is streaming.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

# How often, while waiting for the render's next progress line, to check whether
# the user has said something. Short enough to feel immediate; the check is a lock
# and a length.
POLL_SECONDS = 0.5


async def interleave_checkins(
    source: AsyncIterator,
    has_pending: Callable[[], bool],
    checkin: Callable[[], AsyncIterator],
    poll: float = POLL_SECONDS,
) -> AsyncIterator[tuple[str, object]]:
    """Yield ``("line", item)`` from *source*, and ``("checkin", event)`` from
    ``checkin()`` whenever ``has_pending()`` while *source* is still working.

    *source* keeps running during a check-in — its next item is awaited in a task
    that is not cancelled — so nothing about the render is slowed or reordered.
    A check-in that fails is reported as an event, not raised: a message the agent
    could not read mid-render must not end the render it was sent during.
    """
    agen = source.__aiter__()
    nxt: asyncio.Future | None = None
    try:
        while True:
            if nxt is None:
                nxt = asyncio.ensure_future(agen.__anext__())
            if has_pending():
                try:
                    async for event in checkin():
                        yield "checkin", event
                except Exception as exc:  # noqa: BLE001
                    yield "checkin", {"data": (f"\n\n⚠️ Couldn't read your message while the "
                                               f"render runs ({exc}). The render carries on.")}
            done, _ = await asyncio.wait({nxt}, timeout=poll)
            if not done:
                continue
            finished, nxt = nxt, None
            try:
                item = finished.result()
            except StopAsyncIteration:
                return
            yield "line", item
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()
