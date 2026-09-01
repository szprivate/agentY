"""Letting a turn with no browser behind it see the canvas.

The graph reaches the agent because the PANEL captures it and posts it with the
message. A turn asked for from Slack has no browser round-trip, so it arrived
with no graph, no hooks and no selection — and every canvas tool answered "no
on-canvas graph is loaded this turn", which reads as the agent refusing to look
at a workflow that is plainly open.

So the host asks. A flag rides out on `/agentY/health`, which the panel polls
every five seconds anyway; the panel posts the same payload a message carries;
the snapshot is cached here with the time it was taken. The last part is the one
that matters: a graph handed over as current when it is minutes old is worse
than none, because the agent then edits nodes that have moved.

    python -m unittest discover -s tests
"""

import threading
import time
import unittest

from src.utils import agentY_server as srv


class CanvasRequestTest(unittest.TestCase):

    def setUp(self):
        srv._canvas_cache.clear()
        srv._canvas_wanted_until = 0.0
        self.addCleanup(srv._canvas_cache.clear)

    def _snap(self, prompt=None, hooks=None, selection=None):
        srv.remember_canvas(prompt if prompt is not None else {"3": {"class_type": "KSampler"}},
                            hooks or [], selection or [])

    # ── asking ────────────────────────────────────────────────────────────────
    def test_nobody_is_asking_until_somebody_asks(self):
        self.assertFalse(srv.canvas_wanted())

    def test_a_request_raises_the_flag_the_panel_watches(self):
        done = threading.Event()
        threading.Thread(target=lambda: (srv.request_canvas(wait=1.0), done.set()),
                         daemon=True).start()
        time.sleep(0.2)
        self.assertTrue(srv.canvas_wanted(), "the panel would never know to answer")
        done.wait(3)

    def test_answering_lowers_it_again(self):
        threading.Thread(target=srv.request_canvas, kwargs={"wait": 2.0},
                         daemon=True).start()
        time.sleep(0.2)
        self._snap()
        self.assertFalse(srv.canvas_wanted())

    def test_a_request_that_nobody_answers_stops_asking_eventually(self):
        """Otherwise a panel opened hours later posts a graph for a turn that is
        long gone."""
        self.assertLessEqual(srv._CANVAS_REQUEST_TTL, 120)

    # ── answering ─────────────────────────────────────────────────────────────
    def test_the_answer_reaches_the_waiting_turn(self):
        got = {}
        t = threading.Thread(target=lambda: got.update(srv.request_canvas(wait=3.0)),
                             daemon=True)
        t.start()
        time.sleep(0.3)
        self._snap(prompt={"9": {"class_type": "SaveImage"}})
        t.join(4)
        self.assertEqual(got.get("prompt"), {"9": {"class_type": "SaveImage"}})

    def test_it_does_not_wait_when_one_just_arrived(self):
        """A panel message a moment ago is as good as an answer, and a turn that
        sat for eight seconds for something it already had would be absurd."""
        self._snap()
        started = time.time()
        snap = srv.request_canvas(wait=5.0)
        self.assertLess(time.time() - started, 1.0)
        self.assertTrue(snap.get("prompt"))

    def test_hooks_and_selection_come_along(self):
        self._snap(hooks=[{"hook_node_id": "5"}], selection=[{"id": "3"}])
        snap = srv.request_canvas(wait=0)
        self.assertEqual(snap["hooks"], [{"hook_node_id": "5"}])
        self.assertEqual(snap["selection"], [{"id": "3"}])

    def test_rubbish_in_the_payload_does_not_become_a_graph(self):
        srv.remember_canvas("not a graph", ["not a hook"], "not a list")
        snap = srv.request_canvas(wait=0)
        self.assertIsNone(snap["prompt"])
        self.assertEqual(snap["hooks"], [])
        self.assertEqual(snap["selection"], [])

    # ── the part that keeps it honest ─────────────────────────────────────────
    def test_nothing_at_all_comes_back_empty_rather_than_stale(self):
        self.assertEqual(srv.request_canvas(wait=0), {})

    def test_a_snapshot_too_old_to_trust_is_not_handed_over(self):
        """The agent would edit nodes that have moved, or report on a workflow
        you closed — and nothing on either side would say so."""
        self._snap()
        srv._canvas_cache["ts"] = time.time() - (srv._CANVAS_STALE + 60)
        self.assertEqual(srv.request_canvas(wait=0), {})

    def test_a_recent_one_from_the_last_panel_message_is_good_enough(self):
        self._snap()
        srv._canvas_cache["ts"] = time.time() - 45
        self.assertTrue(srv.request_canvas(wait=0).get("prompt"))

    def test_the_wait_is_short_enough_that_a_dm_still_feels_answered(self):
        self.assertLessEqual(srv._CANVAS_WAIT, 15)


class HealthTest(unittest.TestCase):
    """The flag has to actually reach the panel, on the poll it already makes."""

    def setUp(self):
        srv._canvas_cache.clear()
        srv._canvas_wanted_until = 0.0
        self.addCleanup(srv._canvas_cache.clear)
        from tests.route_client import authorised_client
        app = srv._build_app()
        app.config["TESTING"] = True
        self.client = authorised_client(app)

    def test_health_says_when_the_canvas_is_wanted(self):
        self.assertFalse(self.client.get("/agentY/health").get_json()["want_canvas"])
        srv._canvas_wanted_until = time.time() + 30
        self.assertTrue(self.client.get("/agentY/health").get_json()["want_canvas"])

    def test_the_panel_can_post_one(self):
        r = self.client.post("/agentY/canvas", json={
            "canvas_prompt": {"3": {"class_type": "KSampler"}},
            "canvas_hooks": [{"hook_node_id": "5"}],
            "canvas_selection": [],
        })
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(srv.request_canvas(wait=0)["prompt"],
                         {"3": {"class_type": "KSampler"}})

    def test_a_panel_message_is_itself_a_fresh_snapshot(self):
        """The common case, for free: you type in the panel, then ask from your
        phone a moment later. Without this every Slack turn would wait on a
        round trip for a graph the host was handed seconds ago."""
        from contextlib import ExitStack
        from unittest import mock

        def no_turn(tid, msg, imgs, q, rid, **kw):
            q.put({"type": "done"})
            q.put(None)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(srv, "_run_pipeline_stream", no_turn))
            stack.enter_context(mock.patch.object(
                srv.cs, "get_thread", lambda tid: {"id": tid, "messages": []}))
            stack.enter_context(mock.patch.object(
                srv.cs, "add_message", lambda *a, **k: 1))
            r = self.client.post("/agentY/chat", json={
                "thread_id": "t1", "message": "hello",
                "canvas_prompt": {"7": {"class_type": "LoadImage"}},
                "canvas_hooks": [{"hook_node_id": "9"}],
            })
            r.get_data()
        snap = srv.request_canvas(wait=0)
        self.assertEqual(snap["prompt"], {"7": {"class_type": "LoadImage"}})
        self.assertEqual(snap["hooks"], [{"hook_node_id": "9"}])

    def test_posting_one_answers_the_outstanding_request(self):
        srv._canvas_wanted_until = time.time() + 30
        self.client.post("/agentY/canvas", json={"canvas_prompt": {}})
        self.assertFalse(srv.canvas_wanted())


if __name__ == "__main__":
    unittest.main()
