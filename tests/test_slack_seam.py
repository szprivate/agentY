"""The joins: a Slack message becomes a turn, and a turn becomes Slack messages.

Both halves of this feature test green on their own — the bus fans events out,
the bridge renders them — and neither of those says the two are wired to each
other. That gap is where the last shipped bug in this repo lived (a review halt
stored one id and resumed by looking up another; both sides had tests). So this
drives the real seam: the server's own callbacks, the real bus, the real bridge,
with only Slack itself and the pipeline replaced.

What it is really asking: does a message typed on a phone end up running in the
conversation the panel is in, and does what comes back reach the phone?

    python -m unittest discover -s tests
"""

import queue
import unittest
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import turn_bus
from src.utils.slack_bridge import SlackBridge


class FakeClient:
    def __init__(self):
        self.posted, self.updated, self.uploaded = [], [], []
        self._ts = 0

    def chat_postMessage(self, **kw):
        self._ts += 1
        self.posted.append(kw)
        return {"ts": f"ts{self._ts}"}

    def chat_update(self, **kw):
        self.updated.append(kw)
        return {"ok": True}

    def chat_delete(self, **kw):
        return {"ok": True}

    def files_upload_v2(self, **kw):
        self.uploaded.append(kw)
        return {"ok": True}

    def text(self):
        return " ".join(m.get("text", "") for m in self.posted) \
             + " " + " ".join(m.get("text", "") for m in self.updated)


class SeamTest(unittest.TestCase):

    def setUp(self):
        self.client = FakeClient()
        self.bridge = SlackBridge(client=self.client, allowed_users=["U_ME"],
                                  default_channel="D_ME",
                                  start_turn=srv._slack_start_turn,
                                  answer=srv._slack_answer,
                                  interject=srv._slack_interject)
        self.bridge.bot_user_id = "U_BOT"
        turn_bus.observe(self.bridge.on_turn_event)
        self.addCleanup(turn_bus.unobserve, self.bridge.on_turn_event)
        self.addCleanup(turn_bus._active.clear)
        self.ran = []

        # The pipeline, replaced by something that runs one scripted turn through
        # the REAL bus — which is the piece under test here.
        def fake_stream(thread_id, message, image_paths, q, rid, **kw):
            self.ran.append({"thread_id": thread_id, "message": message,
                             "images": list(image_paths), "rid": rid, **kw})
            t = turn_bus.tee(q, request_id=rid, thread_id=thread_id,
                             origin=kw.get("origin", "panel"), text=message)
            for ev in self.script:
                t.put(ev)
            t.put(None)

        self.script = [{"type": "text", "data": "Rendered it."}, {"type": "done"}]
        self.enterContext(mock.patch.object(srv, "_run_pipeline_stream", fake_stream))
        self.enterContext(mock.patch.object(srv.cs, "add_message", lambda *a, **k: 1))
        self.enterContext(mock.patch.object(srv.cs, "get_thread",
                                            lambda tid: {"id": tid} if tid else None))
        self.enterContext(mock.patch.object(srv.cs, "list_threads",
                                            lambda limit=200: [{"id": "t_recent"}]))
        self.enterContext(mock.patch.object(srv.cs, "create_thread",
                                            lambda **k: "t_new"))

    def _settle(self):
        # What the bridge's worker thread does continuously.
        for _ in range(3):
            self.bridge._tick_turns()
            self.bridge.flush()

    # ── a message from Slack becomes a turn ───────────────────────────────────
    def test_a_dm_runs_a_turn(self):
        out = self.bridge.route("U_ME", "render the hero sheet")
        self.assertEqual(out["action"], "turn")
        self.assertEqual(self.ran[0]["message"], "render the hero sheet")

    def test_it_lands_in_the_conversation_the_panel_is_in(self):
        """Slack is a second window on one session, not a second session."""
        turn_bus._last["thread_id"] = "t_panel"
        self.bridge.route("U_ME", "and now the video")
        self.assertEqual(self.ran[0]["thread_id"], "t_panel")

    def test_with_nothing_run_yet_it_takes_the_most_recent_conversation(self):
        turn_bus._last["thread_id"] = ""
        self.bridge.route("U_ME", "hello")
        self.assertEqual(self.ran[0]["thread_id"], "t_recent")

    def test_the_turn_knows_it_came_from_slack(self):
        self.bridge.route("U_ME", "go")
        self.assertEqual(self.ran[0]["origin"], "slack")

    def test_an_attached_image_reaches_the_pipeline_as_an_input(self):
        self.bridge.route("U_ME", "make this warmer", ["D:/tmp/ref.png"])
        self.assertEqual(self.ran[0]["images"], ["D:/tmp/ref.png"])

    # ── and the turn comes back ───────────────────────────────────────────────
    def test_the_answer_reaches_slack(self):
        self.bridge.route("U_ME", "render it")
        self._settle()
        self.assertIn("Rendered it.", self.client.text())

    def test_a_turn_started_in_the_PANEL_reaches_slack_too(self):
        """The reason this exists: start it at the desk, watch it from a phone."""
        srv._run_pipeline_stream("t_panel", "from the canvas", [], queue.Queue(),
                                 "rid_panel")
        self._settle()
        self.assertIn("Rendered it.", self.client.text())
        self.assertIn("panel", self.client.posted[0]["text"],
                      "it should say the turn came from somewhere else")

    def test_generated_media_is_uploaded(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hero.png"
            p.write_bytes(b"x")
            self.script = [{"type": "output", "path": str(p), "name": "hero.png"},
                           {"type": "done"}]
            self.bridge.route("U_ME", "render it")
            self._settle()
        self.assertEqual(self.client.uploaded[0]["filename"], "hero.png")

    def test_a_turn_that_dies_without_done_still_closes_in_slack(self):
        """Otherwise the phone shows 'working…' forever and nothing says it stopped.

        The queue closes with no `done` on it — the bus makes one up, and what
        matters is the FINAL state of the message, not that an in-progress
        version of it was posted along the way.
        """
        self.script = [{"type": "text", "data": "half a sen"}]   # no done
        self.bridge.route("U_ME", "render it")
        self._settle()
        self.assertTrue(self.client.updated, "the message was never finalised")
        self.assertNotIn("_…_", self.client.updated[-1]["text"])

    # ── answering, from the phone ─────────────────────────────────────────────
    def _asking(self, rid="rid_ask"):
        """A turn parked on a question — it is still running, holding the loop open."""
        turn = turn_bus.Turn(request_id=rid, thread_id="t", origin="slack")
        turn_bus._active[rid] = turn
        self.bridge.on_turn_event({"type": "ask", "request_id": rid,
                                   "prompt": "Retry the failed one?"}, turn)
        self._settle()

    def test_an_agent_question_is_answered_by_the_next_dm(self):
        self._asking()
        self.assertIn("Retry the failed one?", self.client.text())

        # The pipeline's side of an ask: a loop + queue on the reply registry.
        fed = []
        loop = mock.Mock()
        loop.call_soon_threadsafe = lambda fn, val: fed.append(val)
        with mock.patch.dict(srv._reply_registry, {"rid_ask": (loop, mock.Mock())},
                             clear=False):
            out = self.bridge.route("U_ME", "yes, retry")
        self.assertEqual(out["action"], "answer")
        self.assertEqual(fed, ["yes, retry"])

    def test_a_question_whose_turn_already_ended_does_not_swallow_the_next_dm(self):
        """The panel answered it, or the turn died. Either way the question is
        gone, and treating the next message as its answer would drop it."""
        self._asking()
        turn = turn_bus._active.pop("rid_ask")
        self.bridge.on_turn_event({"type": "done"}, turn)
        out = self.bridge.route("U_ME", "make another one")
        self.assertEqual(out["action"], "turn")

    def test_a_dm_during_a_running_turn_is_interjected(self):
        turn_bus._active["rid_live"] = turn_bus.Turn(request_id="rid_live",
                                                     thread_id="t")
        self.bridge.on_turn_event({"type": "text", "data": "…"},
                                  turn_bus._active["rid_live"])
        with mock.patch("src.utils.interject_bus.post", return_value=True) as post:
            out = self.bridge.route("U_ME", "actually, make it warmer")
        self.assertEqual(out["action"], "interject")
        post.assert_called_once()
        self.assertEqual(self.ran, [], "it must not have started a second turn")


if __name__ == "__main__":
    unittest.main()
