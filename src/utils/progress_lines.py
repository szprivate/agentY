"""Progress lines, turned into chunks of the assistant's text stream.

A tool running mid-turn pushes status into
:mod:`agenty_core.utils.progress_signal` — ``"🚀 Queuing iteration 3/8…"`` — and
whoever is draining it hands each one to the panel as a ``{"data": …}`` event.
That is the same event the model's own tokens arrive on, and it is a stream of
**chunks**: the panel concatenates them exactly as they come, with nothing in
between.

A line is not a chunk. Eight queue lines pushed back to back reached the panel as
one run-on sentence — *"…prompt_id=…🚀 Queuing iteration 2/8…✅ Iteration 2/8
queued…"* — because each was appended precisely where the last one ended. The
end-of-turn executor never showed this: that path yields ``f"\\n{line}"`` by
hand. The run-now path pushes through this buffer instead, and pushed nothing to
separate the lines with.

So the conversion lives here rather than at each drain site: a line reaching the
stream begins its own line, once, however it was pushed and whoever drains it.

Consumers that classify a chunk by its opening characters (the panel routes
``"⬇️ "`` download bars to their own channel) must therefore look past leading
whitespace — see ``_translate`` in ``agentY_server``.
"""

from __future__ import annotations

from agenty_core.utils.progress_signal import drain as _drain


def as_chunk(line: str) -> str:
    """One pushed progress line, as a chunk that starts on its own line."""
    text = str(line)
    return text if text.startswith("\n") else "\n" + text


def drain_chunks() -> list[str]:
    """Drain the progress buffer, each line ready to yield as ``{"data": …}``.

    Atomic, because :func:`agenty_core.utils.progress_signal.drain` is: two
    consumers (the pipeline's own loop and the server's flush pump) race for
    this buffer and neither may see a line twice.
    """
    return [as_chunk(line) for line in _drain()]
