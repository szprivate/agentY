"""Retry a model call that a provider refused for a reason that passes.

DashScope answers some failures on its own side as a 400
``invalid_parameter_error`` — "Download multimodal file timed out" is one: its
model worker could not fetch an image from DashScope's own storage in time. A
400 is not retried by the OpenAI client or by Strands, so one slow fetch ended
the user's turn with the raw ``data: {...}`` line. The same request usually
goes through a few seconds later.

:class:`TransientRetryMixin` goes in front of an ``OpenAIModel``. It retries
the call only while nothing has been streamed yet (so no half answer is ever
repeated) and only for the messages in :data:`TRANSIENT_MARKERS`.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Lower-case fragments of provider errors that are worth sending again as-is.
TRANSIENT_MARKERS = (
    "download multimodal file timed out",
    "download the media resource timed out",
)

# Seconds to wait before each retry.
RETRY_DELAYS = (2.0, 6.0)


def is_transient(exc: BaseException) -> bool:
    text = str(exc).lower()
    body = getattr(exc, "body", None)
    if body is not None:
        text += " " + str(body).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


class TransientRetryMixin:
    """Put before ``OpenAIModel`` (or a subclass of it)."""

    retry_delays = RETRY_DELAYS

    async def stream(self, *args, **kwargs):  # type: ignore[override]
        attempt = 0
        while True:
            started = False
            try:
                async for event in super().stream(*args, **kwargs):  # type: ignore[misc]
                    started = True
                    yield event
                return
            except Exception as exc:  # noqa: BLE001 - re-raised unless transient
                if started or attempt >= len(self.retry_delays) or not is_transient(exc):
                    raise
                delay = self.retry_delays[attempt]
                attempt += 1
                logger.warning("model call failed (%s); retry %d/%d in %.0fs",
                               str(exc)[:160], attempt, len(self.retry_delays), delay)
                print(f"[agentY] provider error ({str(exc)[:120]}); "
                      f"retrying {attempt}/{len(self.retry_delays)} in {delay:.0f}s")
                await asyncio.sleep(delay)
