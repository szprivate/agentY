"""The server end of the team features: what the host does with a channel request,
a queued API request and the deep health check.

The pipeline is replaced; the host's own functions are real.

    python -m unittest tests.test_server_seam
"""

import os
import queue
import tempfile
import unittest
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import server_mode, team_store, turn_bus, turn_queue, user_auth


class _Base(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.enterContext(mock.patch.dict(os.environ, {"AGENTY_TEAM_DIR": d.name}))
        team_store._requesters.clear()
        team_store._loaded = False
        self.addCleanup(team_store._requesters.clear)
        self.ran = []
        self.bound = {}
        self.messages = []

        def fake_stream(thread_id, message, image_paths, q, rid, **kw):
            self.ran.append({"thread_id": thread_id, "message": message, "rid": rid, **kw})
            q.put({"type": "done"})
            q.put(None)

        self.enterContext(mock.patch.object(srv, "_run_pipeline_stream", fake_stream))
        self.enterContext(mock.patch.object(srv.cs, "add_message",
                                            lambda tid, role, text: self.messages.append(text)))
        self.enterContext(mock.patch.object(srv.cs, "get_thread",
                                            lambda tid: {"id": tid} if tid else None))
        self.enterContext(mock.patch.object(srv.cs, "create_thread", lambda **k: "t_new"))
        self.enterContext(mock.patch.object(
            srv.cs, "set_slack_thread", lambda tid, ch, ts: self.bound.__setitem__(tid, (ch, ts))))
        self.enterContext(mock.patch.object(
            srv.cs, "get_slack_thread",
            lambda tid: ({"channel": self.bound[tid][0], "root_ts": self.bound[tid][1]}
                         if tid in self.bound else None)))
        self.enterContext(mock.patch.object(srv, "request_canvas", lambda wait=0: {}))
        self.enterContext(mock.patch.object(srv, "_drain_queue", lambda q: None))


class SlackChannelTurnTest(_Base):

    def _start(self, **kw):
        rid = srv._slack_start_turn("make a poster", [], "", **kw)
        for t in __import__("threading").enumerate():
            if t.name == "agentY-slack-turn":
                t.join(2)
        return rid

    def test_a_channel_request_is_bound_to_its_thread_and_owned(self):
        self._start(requester={"user": "U1", "channel": "C1", "project": "C1", "name": "Anna"},
                    slack_root=("C1", "100.1"))
        self.assertEqual(self.bound["t_new"], ("C1", "100.1"))
        self.assertEqual(team_store.requester("t_new")["user"], "U1")

    def test_the_agent_is_told_who_asked_and_what_to_remember(self):
        team_store.add_note("users", "U1", "always 16:9")
        self._start(requester={"user": "U1", "channel": "C1", "project": "C1", "name": "Anna"},
                    slack_root=("C1", "100.1"))
        sent = self.ran[0]["message"]
        self.assertIn("REQUESTED BY Anna", sent)
        self.assertIn("always 16:9", sent)
        self.assertTrue(sent.endswith("make a poster"))
        self.assertEqual(self.messages, ["make a poster"])        # the chat log keeps the words only

    def test_a_dm_turn_is_unchanged(self):
        self._start()
        self.assertEqual(self.ran[0]["message"], "make a poster")
        self.assertEqual(self.bound, {})
        self.assertIs(self.ran[0]["canvas_provider"], srv.request_canvas)

    def test_a_headless_turn_asks_for_no_canvas(self):
        with mock.patch.object(server_mode, "headless", return_value=True):
            self._start()
        self.assertIsNone(self.ran[0]["canvas_provider"])


class ApiQueueTest(_Base):

    def setUp(self):
        super().setUp()
        self.q = turn_queue.TurnQueue(limit=1, max_queued=1)
        self.enterContext(mock.patch.object(turn_queue, "_SHARED", self.q))
        srv._api_tickets.clear()
        self.addCleanup(srv._api_tickets.clear)

    def _wait_threads(self):
        for t in __import__("threading").enumerate():
            if t.name == "agentY-api-turn":
                t.join(2)

    def test_a_desktop_host_starts_at_once_and_never_queues(self):
        with mock.patch.object(server_mode, "headless", return_value=False), \
                mock.patch.object(user_auth, "enabled", return_value=False):
            srv._api_start_turn("a", [], "")
            srv._api_start_turn("b", [], "")
            self._wait_threads()
        self.assertEqual(len(self.ran), 2)
        self.assertEqual(self.q.snapshot()["running"], [])

    def test_a_server_host_queues_the_second_request(self):
        with mock.patch.object(server_mode, "headless", return_value=True):
            rid1, _ = srv._api_start_turn("a", [], "")
            rid2, _ = srv._api_start_turn("b", [], "")
            self._wait_threads()
        self.assertEqual(len(self.ran), 1)
        self.assertEqual(srv._api_tickets[rid2].state, turn_queue.QUEUED)
        self.assertEqual(self.q.position(srv._api_tickets[rid2]), 1)
        self.q.finished(rid1)
        self._wait_threads()
        self.assertEqual([r["message"] for r in self.ran], ["a", "b"])

    def test_the_queue_limit_is_an_error_the_route_can_report(self):
        with mock.patch.object(server_mode, "headless", return_value=True):
            srv._api_start_turn("a", [], "")
            srv._api_start_turn("b", [], "")
            with self.assertRaises(ValueError) as cm:
                srv._api_start_turn("c", [], "")
        self.assertIn("queue is full", str(cm.exception))

    def test_the_users_turn_is_theirs(self):
        with mock.patch.object(user_auth, "enabled", return_value=True):
            _, tid = srv._api_start_turn("a", [], "", user={"name": "anna", "roles": []})
            self._wait_threads()
        self.assertEqual(team_store.requester(tid)["user"], "anna")
        self.assertIn("REQUESTED BY anna", self.ran[0]["message"])


class RoutesTest(unittest.TestCase):
    """The HTTP surface: queue listing, queued status and deep health."""

    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        self.enterContext(mock.patch.object(srv, "_guard_verdict", lambda req: (True, "")))
        self.client = srv._build_app().test_client()

    def test_queue_endpoint_lists_who_waits(self):
        q = turn_queue.TurnQueue(limit=1)
        q.submit(lambda: "r1", user="anna", text="poster")
        q.submit(lambda: "r2", user="ben", text="logo")
        with mock.patch.object(turn_queue, "_SHARED", q):
            body = self.client.get("/agentY/queue").get_json()
        self.assertEqual(body["running"][0]["user"], "anna")
        self.assertEqual(body["queued"][0]["user"], "ben")
        self.assertEqual(body["queued"][0]["position"], 1)

    def test_default_health_is_unchanged(self):
        body = self.client.get("/agentY/health").get_json()
        self.assertNotIn("server", body)
        self.assertEqual(body["status"], "ok")

    def test_deep_health_reports_mode_queue_slack_and_comfyui(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            body = self.client.get("/agentY/health?deep=1").get_json()
        s = body["server"]
        self.assertEqual(set(s) >= {"headless", "user_tokens", "queue", "slack", "comfyui"}, True)
        self.assertFalse(s["comfyui"]["ok"])
        self.assertEqual(body["status"], "ok")      # a dead ComfyUI is reported, not fatal

    def test_a_queued_turn_has_a_status(self):
        q = turn_queue.TurnQueue(limit=1)
        q.submit(lambda: "r1")
        t2 = q.submit(lambda: "r2")
        srv._api_tickets["r2"] = t2
        self.addCleanup(srv._api_tickets.clear)
        with mock.patch.object(turn_queue, "_SHARED", q):
            body = self.client.get("/agentY/turns/r2").get_json()
        self.assertEqual((body["status"], body["position"]), ("queued", 1))


if __name__ == "__main__":
    unittest.main()
