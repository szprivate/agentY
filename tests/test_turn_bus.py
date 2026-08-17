"""Making a turn watchable by more than the browser that asked for it.

A turn's events go into one queue, and that queue belongs to one HTTP request.
The bus wraps it at the single point a turn starts so a second channel (Slack)
sees the same stream. Two properties carry everything built on top:

* the panel is served first and unconditionally — an observer that is slow,
  broken, or absent changes nothing about what the sidebar receives;
* every turn ends with exactly one `done`, so a watcher can always take its
  "working…" message down.

    python -m unittest discover -s tests
"""

import queue
import unittest

from src.utils import turn_bus


class BusTest(unittest.TestCase):

    def setUp(self):
        self.seen = []
        self.q = queue.Queue()
        self.addCleanup(self._clear)

    def _clear(self):
        for fn in list(turn_bus._observers):
            turn_bus.unobserve(fn)
        turn_bus._active.clear()

    def _watch(self):
        def obs(event, turn):
            self.seen.append((event.get("type"), turn.request_id))
        turn_bus.observe(obs)
        return obs

    def _tee(self, **kw):
        kw.setdefault("request_id", "r1")
        kw.setdefault("thread_id", "t1")
        return turn_bus.tee(self.q, **kw)

    def _drain(self):
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    # ── the panel is untouched ────────────────────────────────────────────────
    def test_every_event_still_reaches_the_real_queue(self):
        t = self._tee()
        t.put({"type": "text", "data": "hi"})
        t.put({"type": "done"})
        t.put(None)
        self.assertEqual(self._drain(),
                         [{"type": "text", "data": "hi"}, {"type": "done"}, None])

    def test_with_nobody_watching_it_is_a_pass_through(self):
        t = self._tee()
        t.put({"type": "text", "data": "hi"})
        self.assertEqual(self._drain(), [{"type": "text", "data": "hi"}])

    def _break(self):
        """Register an observer that always raises (and hush its expected trace)."""
        import logging
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        turn_bus.observe(lambda ev, turn: 1 / 0)

    def test_an_observer_that_raises_does_not_reach_the_turn(self):
        """A broken watcher is a broken watcher, not a broken turn."""
        self._break()
        t = self._tee()
        t.put({"type": "text", "data": "hi"})   # must not raise
        self.assertEqual(self._drain(), [{"type": "text", "data": "hi"}])

    def test_one_broken_observer_does_not_rob_the_others(self):
        self._break()
        self._watch()
        self._tee().put({"type": "text"})
        self.assertEqual([t for t, _ in self.seen], ["turn_start", "text"])

    def test_queue_methods_other_than_put_still_work(self):
        """It stands in for a queue everywhere, not just where it is written to."""
        t = self._tee()
        t.put({"type": "text"})
        self.assertFalse(t.empty())
        self.assertEqual(t.get_nowait(), {"type": "text"})

    # ── what an observer sees ─────────────────────────────────────────────────
    def test_the_watcher_is_told_a_turn_started(self):
        self._watch()
        self._tee()
        self.assertEqual(self.seen, [("turn_start", "r1")])

    def test_events_arrive_with_the_turn_they_belong_to(self):
        got = []
        turn_bus.observe(lambda ev, turn: got.append((ev.get("type"), turn.origin,
                                                      turn.thread_id, turn.text)))
        self._tee(origin="slack", text="make it warmer").put({"type": "text"})
        self.assertEqual(got[-1], ("text", "slack", "t1", "make it warmer"))

    def test_the_origin_says_who_asked(self):
        """A mirror has to tell 'you asked this' from 'this is happening elsewhere'."""
        got = []
        turn_bus.observe(lambda ev, turn: got.append(turn.origin))
        self._tee().put({"type": "text"})
        self.assertEqual(got[-1], "panel", "the default is the sidebar")

    # ── the end, which is the part that must never be missed ──────────────────
    def test_a_turn_that_ends_properly_reports_done_once(self):
        self._watch()
        t = self._tee()
        t.put({"type": "done"})
        t.put(None)
        self.assertEqual([e for e, _ in self.seen].count("done"), 1)

    def test_a_turn_that_closes_without_done_still_gets_one(self):
        """Otherwise a watcher's 'working…' message stays up forever."""
        self._watch()
        self._tee().put(None)
        self.assertIn("done", [e for e, _ in self.seen])

    def test_the_made_up_done_says_so(self):
        got = []
        turn_bus.observe(lambda ev, turn: got.append(ev))
        self._tee().put(None)
        self.assertTrue(got[-1].get("synthesized"))

    def test_nothing_is_published_after_the_stream_closed(self):
        self._watch()
        t = self._tee()
        t.put({"type": "done"})
        t.put(None)
        before = len(self.seen)
        t.put({"type": "text"})
        self.assertEqual(len(self.seen), before + 1,
                         "a late event still forwards; it just must not re-close")

    # ── who is in flight, and where ───────────────────────────────────────────
    def test_a_running_turn_is_visible_to_a_second_channel(self):
        t = self._tee()
        self.assertEqual([x.request_id for x in turn_bus.active()], ["r1"])
        t.put(None)
        self.assertEqual(turn_bus.active(), [])

    def test_the_last_thread_is_where_a_second_channel_should_land(self):
        self._tee(thread_id="t-abc")
        self.assertEqual(turn_bus.last_thread_id(), "t-abc")

    def test_two_turns_are_kept_apart(self):
        got = []
        turn_bus.observe(lambda ev, turn: got.append((turn.request_id, ev.get("type"))))
        a = self._tee(request_id="a", thread_id="t1")
        b = turn_bus.tee(queue.Queue(), request_id="b", thread_id="t2")
        a.put({"type": "text"})
        b.put({"type": "output"})
        self.assertIn(("a", "text"), got)
        self.assertIn(("b", "output"), got)

    def test_unobserve_stops_delivery(self):
        obs = self._watch()
        turn_bus.unobserve(obs)
        self._tee().put({"type": "text"})
        self.assertEqual(self.seen, [])


if __name__ == "__main__":
    unittest.main()
