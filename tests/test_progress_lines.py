"""Status lines pushed mid-tool, arriving in the panel as lines.

A tool that runs the graph itself — `run_workflow_now`, `apply_canvas_hooks`
with `run_now`, `iterate_step` — reports its progress by pushing lines into the
progress buffer, because the pipeline's own loop is blocked awaiting the tool
and cannot yield anything meanwhile. Those lines reach the panel on the same
event the model's tokens do, and that event is a stream of CHUNKS: appended
exactly where the last one ended, with nothing in between.

So eight queue lines arrived as one paragraph::

    …prompt_id=b765a975🚀 Queuing iteration 2/8…✅ Iteration 2/8 queued…

The end-of-turn executor never showed it — that path wrote its own newline by
hand — which is why the run-now path was the one that read as a wall of text.

    python -m unittest discover -s tests
"""

import asyncio
import queue
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils.progress_signal import drain as raw_drain
from agenty_core.utils.progress_signal import push as push_progress

from src.utils import agentY_server as srv
from src.utils.progress_lines import as_chunk, drain_chunks


class AsChunkTest(unittest.TestCase):
    """A line is not a chunk; this is the conversion."""

    def setUp(self):
        raw_drain()  # anything a previous test left behind
        self.addCleanup(raw_drain)

    def test_a_line_begins_its_own_line(self):
        self.assertEqual(as_chunk("🚀 Queuing iteration 1/8…"),
                         "\n🚀 Queuing iteration 1/8…")

    def test_two_lines_do_not_run_together(self):
        push_progress("🚀 Queuing iteration 1/8…")
        push_progress("✅ Iteration 1/8 queued")
        joined = "".join(drain_chunks())
        self.assertEqual(len(joined.strip().splitlines()), 2, joined)

    def test_a_line_that_already_breaks_is_not_broken_twice(self):
        """Otherwise the paths that write their own newline gain a blank line."""
        self.assertEqual(as_chunk("\n✅ Done."), "\n✅ Done.")

    def test_nothing_pushed_drains_to_nothing(self):
        self.assertEqual(drain_chunks(), [])

    def test_draining_takes_the_lines_with_it(self):
        """Two consumers race for this buffer — the pipeline's loop and the
        server's flush pump — and a line delivered twice is a line shown twice."""
        push_progress("🚀 one")
        self.assertEqual(len(drain_chunks()), 1)
        self.assertEqual(drain_chunks(), [])


# ── the seam: a turn, its progress, and what the panel is handed ──────────────

class _FakePipeline:
    """Just enough Pipeline for one turn of ``_run_pipeline_turn``."""

    def __init__(self, script):
        self.script = script
        self._session = SimpleNamespace(current_output_paths=[],
                                        last_user_input_images=[], session_id="t")
        self._last_brainbriefing_json = None
        self._last_prior_summary = None

    async def stream_async(self, content, **kw):
        for kind, text in self.script:
            if kind == "push":
                push_progress(text)
            elif kind == "wait":
                await asyncio.sleep(float(text))
            else:
                yield {"data": text}

    async def _await_pending_compression(self):
        return None


def _turn(case, script):
    """Run one turn and hand back the events the panel would receive."""
    out_q: "queue.Queue" = queue.Queue()
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(srv, "_agent_ref", _FakePipeline(script)))
        stack.enter_context(mock.patch.object(srv, "_restore_state", lambda *a, **k: None))
        stack.enter_context(mock.patch.object(srv, "_save_state", lambda *a, **k: None))
        stack.enter_context(mock.patch.object(srv, "_resolve_qa_briefing",
                                              lambda *a, **k: None))
        stack.enter_context(mock.patch.object(
            srv.cs, "get_thread",
            lambda tid: {"id": tid, "messages": [{"role": "assistant", "content": "hi"}]}))
        stack.enter_context(mock.patch.object(srv.cs, "add_message", lambda *a, **k: 1))
        srv._run_pipeline_turn("t1", "go", [], out_q, "rid1", {"emitted": False})
    events = []
    while True:
        ev = out_q.get_nowait()
        if ev is None:
            break
        events.append(ev)
    return events


def _said(events):
    """The assistant text, as the panel concatenates it."""
    return "".join(e.get("data", "") for e in events if e.get("type") == "text")


def _status(events):
    """The status lines, one per progress event, as the run card receives them."""
    return [e["data"] for e in events if e.get("type") == "progress"]


class ThroughTheTurnTest(unittest.TestCase):
    """Executor lines are status, not reply. They used to ride the text channel,
    so a ten-image batch appended forty lines to the answer and the transcript
    kept them all under the agent's name; the panel's run card never saw one."""

    def setUp(self):
        raw_drain()
        self.addCleanup(raw_drain)

    def test_the_executor_lines_arrive_as_status_one_per_line(self):
        events = _turn(self, [
            ("push", "🚀 Queuing iteration 1/8…"),
            ("push", "✅ Iteration 1/8 queued · prompt_id=`b765a975`"),
            ("push", "🚀 Queuing iteration 2/8…"),
            ("push", "⏳ All 8 workflow(s) queued — monitoring concurrently…"),
        ])
        self.assertEqual(_status(events), [
            "🚀 Queuing iteration 1/8…",
            "✅ Iteration 1/8 queued · prompt_id=`b765a975`",
            "🚀 Queuing iteration 2/8…",
            "⏳ All 8 workflow(s) queued — monitoring concurrently…",
        ])

    def test_they_stay_out_of_the_reply(self):
        said = _said(_turn(self, [("say", "Running the stage now."),
                                  ("wait", "0.35"),
                                  ("push", "[2/8] ⏳ Queue: 4 job(s) ahead"),
                                  ("say", " Done.")]))
        self.assertNotIn("Queue", said)
        self.assertEqual(said, "Running the stage now. Done.")

    def test_the_pump_carries_them_live_rather_than_at_the_end(self):
        """Blocked in a tool, the buffer is the only thing streaming — a batch
        that only appeared once the turn finished would look like a hang."""
        events = _turn(self, [("push", "🚀 Queuing iteration 1/8…"),
                              ("wait", "0.35"),
                              ("say", "Queued them all.")])
        kinds = [e["type"] for e in events if e.get("type") in ("progress", "text")]
        self.assertEqual(kinds, ["progress", "text"])

    def test_a_status_line_carries_no_separator(self):
        """The newline that kept a line off the previous one is text-channel
        plumbing; a status event is already its own line."""
        self.assertEqual(_status(_turn(self, [("push", "🎯 Hook scope: 14 node(s)…")])),
                         ["🎯 Hook scope: 14 node(s)…"])


class DownloadBarsKeepTheirChannelTest(unittest.TestCase):
    """A `⬇️` bar is replaced in place, not appended: one line, redrawn. It is
    recognised by its opening characters, so a leading newline must not hide it
    — a 100-frame download would otherwise unroll into the transcript."""

    def setUp(self):
        raw_drain()
        self.addCleanup(raw_drain)

    def test_a_pushed_download_bar_is_still_progress(self):
        events = _turn(self, [("push", "⬇️ [████░░░░░░] 45% …")])
        self.assertEqual([e["type"] for e in events if e["type"] in ("progress", "text")],
                         ["progress"])

    def test_and_it_is_not_shown_with_its_separator(self):
        events = _turn(self, [("push", "⬇️ [████░░░░░░] 45% …")])
        bar = next(e for e in events if e["type"] == "progress")
        self.assertEqual(bar["data"], "⬇️ [████░░░░░░] 45% …")

    def test_a_comfyui_console_line_still_has_its_own_channel(self):
        events = _turn(self, [("push", "🖥 loading model …")])
        self.assertEqual([e["type"] for e in events if e["type"] in ("console", "text")],
                         ["console"])


if __name__ == "__main__":
    unittest.main()
