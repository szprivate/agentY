"""Asking the open ComfyUI page something, and waiting for its answer.

Everything else crossing this boundary goes one way: `canvas_patch` pushes edits
at the panel and never hears back, and the panel posts its graph up with each
message without asking anything. Two things live only in the browser, though —
what the graph LOOKS like, and which other workflows are open in ComfyUI's tabs —
and both need a round trip.

Not over SSE, and the reason is the trap this module exists to avoid: the
canvas-patch drain runs INSIDE the orchestrator's `stream_async` loop, so a tool
that blocked waiting for an answer would be holding the very loop meant to
deliver its question. Hence a separate long-poll.

    python -m unittest discover -s tests
"""

import threading
import time
import unittest

from src.utils import canvas_probe


class RendezvousTest(unittest.TestCase):
    """One waiter, one answer, and every way that can fail to happen."""

    def setUp(self):
        canvas_probe.clear()

    tearDown = setUp

    def _answer(self, data, delay=0.05, kind=None):
        """Play the panel: poll, then reply. Returns the thread."""
        def run():
            deadline = time.time() + 3
            while time.time() < deadline:
                probe = canvas_probe.take()
                if probe and (kind is None or probe["kind"] == kind):
                    time.sleep(delay)
                    canvas_probe.reply(probe["probe_id"], data)
                    return
                time.sleep(0.01)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def test_the_answer_comes_back_to_the_waiter(self):
        self._answer({"data_url": "data:image/png;base64,AAAA", "nodes": 10})
        reply = canvas_probe.request("screenshot", timeout=3)
        self.assertEqual(reply["nodes"], 10)
        self.assertNotIn("error", reply)

    def test_the_request_reaches_the_panel_intact(self):
        seen = {}

        def run():
            deadline = time.time() + 3
            while time.time() < deadline:
                probe = canvas_probe.take()
                if probe:
                    seen.update(probe)
                    canvas_probe.reply(probe["probe_id"], {"ok": True})
                    return
                time.sleep(0.01)

        threading.Thread(target=run, daemon=True).start()
        canvas_probe.request("screenshot", {"maxScale": 2}, timeout=3)
        self.assertEqual(seen.get("kind"), "screenshot")
        self.assertEqual(seen.get("payload"), {"maxScale": 2})
        self.assertTrue(seen.get("probe_id"))

    def test_a_page_that_never_answers_times_out_rather_than_hanging(self):
        """A closed tab. The ordinary case, and it must not hold the turn."""
        started = time.time()
        reply = canvas_probe.request("screenshot", timeout=0.5)
        self.assertTrue(reply.get("timeout"))
        self.assertIn("closed", reply["error"])
        self.assertLess(time.time() - started, 2.0, "the wait ran long past its timeout")

    def test_nothing_is_left_waiting_after_a_timeout(self):
        canvas_probe.request("screenshot", timeout=0.5)
        self.assertEqual(canvas_probe.pending_count(), 0)
        self.assertIsNone(canvas_probe.take(), "a timed-out probe was still on offer")

    def test_a_late_answer_is_refused_rather_than_lost_silently(self):
        canvas_probe.request("screenshot", timeout=0.5)
        self.assertFalse(canvas_probe.reply("whatever", {"ok": True}))

    def test_take_returns_nothing_when_nobody_is_asking(self):
        self.assertIsNone(canvas_probe.take())

    def test_two_probes_do_not_get_each_others_answers(self):
        replies = {}

        def ask(kind):
            replies[kind] = canvas_probe.request(kind, timeout=3)

        threads = [threading.Thread(target=ask, args=(k,), daemon=True)
                   for k in ("screenshot", "open_workflows")]
        for t in threads:
            t.start()
        # Wait until BOTH are actually registered before answering either: answer
        # too early and the second thread is still starting, which fails the test
        # for a reason that has nothing to do with routing.
        deadline = time.time() + 3
        while canvas_probe.pending_count() < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(canvas_probe.pending_count(), 2)
        # Answer each with something only it should receive.
        answered = 0
        while answered < 2 and time.time() < deadline:
            probe = canvas_probe.take()
            if probe:
                canvas_probe.reply(probe["probe_id"], {"echo": probe["kind"]})
                answered += 1
            else:
                time.sleep(0.01)
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(replies["screenshot"]["echo"], "screenshot")
        self.assertEqual(replies["open_workflows"]["echo"], "open_workflows")

    def test_a_probe_handed_out_is_not_handed_out_twice(self):
        """Two panels (or a duplicated poll) must not both answer one question."""
        threading.Thread(
            target=lambda: canvas_probe.request("screenshot", timeout=1),
            daemon=True).start()
        time.sleep(0.05)
        first = canvas_probe.take()
        self.assertIsNotNone(first)
        self.assertIsNone(canvas_probe.take(), "the same probe was served twice")

    def test_an_answered_probe_is_never_offered_again(self):
        """The bug this file caught on its first run.

        A probe stays in the pending map until its waiter wakes up and cleans up.
        In that window `take()` would hand the panel a question that already had
        its answer — and, worse, the probe genuinely waiting behind it was then
        never served at all, so a second screenshot request could time out while
        the panel sat there re-answering the first.
        """
        threading.Thread(
            target=lambda: canvas_probe.request("screenshot", timeout=3),
            daemon=True).start()
        threading.Thread(
            target=lambda: canvas_probe.request("open_workflows", timeout=3),
            daemon=True).start()
        deadline = time.time() + 3
        while canvas_probe.pending_count() < 2 and time.time() < deadline:
            time.sleep(0.01)

        first = canvas_probe.take()
        canvas_probe.reply(first["probe_id"], {"ok": True})
        # Answered, but its waiter has not necessarily cleaned up yet.
        second = canvas_probe.take()
        self.assertIsNotNone(second, "the waiting probe was never served")
        self.assertNotEqual(second["probe_id"], first["probe_id"],
                            "an answered probe was handed out a second time")
        self.assertFalse(canvas_probe.reply(first["probe_id"], {"ok": True}),
                         "an answered probe accepted a second answer")

    def test_a_reload_mid_probe_can_pick_the_question_up_again(self):
        """The page took the question and died before answering.

        The waiter is still blocked, so the probe has to become available again —
        otherwise a reload at the wrong moment costs the user their screenshot for
        the whole timeout.
        """
        threading.Thread(
            target=lambda: canvas_probe.request("screenshot", timeout=5),
            daemon=True).start()
        time.sleep(0.05)
        self.assertIsNotNone(canvas_probe.take())
        canvas_probe._RESERVE_SECONDS = 0.2      # what a reload's delay looks like
        try:
            time.sleep(0.35)
            again = canvas_probe.take()
        finally:
            canvas_probe._RESERVE_SECONDS = 6.0
        self.assertIsNotNone(again, "a reloaded page could never retry the probe")

    def test_clear_releases_anyone_waiting(self):
        """A host shutting down must not leave a thread blocked on a dead page."""
        done = threading.Event()

        def ask():
            canvas_probe.request("screenshot", timeout=10)
            done.set()

        threading.Thread(target=ask, daemon=True).start()
        time.sleep(0.05)
        canvas_probe.clear()
        self.assertTrue(done.wait(timeout=2), "clear() left a waiter blocked")


class OpenWorkflowsBlockTest(unittest.TestCase):
    """What the agent is told about ComfyUI's tabs."""

    def _tabs(self, *rows):
        return canvas_probe.describe_open_workflows(list(rows))

    def test_one_workflow_says_nothing_at_all(self):
        """Everybody already assumes this; a block saying so is rent per turn."""
        self.assertEqual(self._tabs({"name": "a", "active": True}), "")
        self.assertEqual(self._tabs(), "")

    def test_several_workflows_are_listed_with_the_active_one_marked(self):
        block = self._tabs(
            {"name": "refs", "active": False},
            {"name": "video", "active": True, "nodes": 88})
        self.assertIn("2 workflows are open", block)
        self.assertIn("88 nodes", block)
        # The mark has to be on the RIGHT row: the header says the word ACTIVE
        # too, so merely finding it in the block proves nothing about which
        # workflow the agent thinks it is holding.
        rows = {ln.split("- ", 1)[1].split("  [")[0]: ln
                for ln in block.splitlines() if ln.startswith("  - ")}
        self.assertIn("ACTIVE", rows["video"])
        self.assertNotIn("ACTIVE", rows["refs"])

    def test_it_says_the_others_cannot_be_touched(self):
        """The correction that matters: the agent must ask, not switch tabs."""
        block = self._tabs({"name": "a", "active": True}, {"name": "b"})
        self.assertIn("Only the active tab", block)
        self.assertIn("ask them to click", block)
        self.assertIn("do not try to open it yourself", block)

    def test_unsaved_work_is_flagged(self):
        block = self._tabs({"name": "a", "active": True},
                           {"name": "b", "modified": True})
        self.assertIn("unsaved changes", block)

    def test_a_nameless_tab_still_gets_a_line(self):
        block = self._tabs({"active": True}, {"name": ""})
        self.assertEqual(block.count("  - "), 2)
        self.assertIn("untitled", block)

    def test_junk_rows_are_ignored_rather_than_crashing(self):
        self.assertEqual(canvas_probe.describe_open_workflows(
            ["not a dict", None, 42]), "")
        block = canvas_probe.describe_open_workflows(
            [{"name": "a", "active": True}, "junk", {"name": "b"}])
        self.assertIn("2 workflows are open", block)


if __name__ == "__main__":
    unittest.main()
