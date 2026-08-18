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
        # The Slack side of a conversation: which thread is which. Bound in the
        # store for real; stubbed here so the seam under test is the routing.
        self.bound = {}
        self.enterContext(mock.patch.object(
            srv.cs, "set_slack_thread",
            lambda tid, ch, ts: self.bound.__setitem__((ch, ts), tid)))
        self.enterContext(mock.patch.object(
            srv.cs, "get_slack_thread",
            lambda tid: next(({"channel": c, "root_ts": s}
                              for (c, s), v in self.bound.items() if v == tid), None)))
        self.enterContext(mock.patch(
            "src.utils.slack_bridge.cs.thread_for_slack",
            lambda ch, ts: self.bound.get((ch, ts))))
        self.enterContext(mock.patch(
            "src.utils.slack_bridge.cs.get_slack_thread",
            lambda tid: next(({"channel": c, "root_ts": s}
                              for (c, s), v in self.bound.items() if v == tid), None)))
        self.enterContext(mock.patch(
            "src.utils.slack_bridge.cs.set_slack_thread",
            lambda tid, ch, ts: self.bound.__setitem__((ch, ts), tid)))
        self.enterContext(mock.patch(
            "src.utils.slack_bridge.cs.list_threads",
            lambda limit=200: [{"id": "t_new", "title": "New chat"}]))

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

    def test_a_top_level_dm_starts_a_NEW_conversation(self):
        """Conversations are Slack threads now: the top level is where a new one
        begins, the way a new chat does in the panel."""
        turn_bus._last["thread_id"] = "t_panel"
        self.bridge.route("U_ME", "and now the video")
        self.assertEqual(self.ran[0]["thread_id"], "t_new",
                         "it joined an existing conversation instead of starting one")

    def test_replying_in_a_thread_goes_back_to_THAT_conversation(self):
        self.bound[("D_ME", "root_abc")] = "t_earlier"
        self.bridge.route("U_ME", "warmer", thread_ts="root_abc")
        self.assertEqual(self.ran[0]["thread_id"], "t_earlier")

    def test_a_reply_in_an_unknown_thread_starts_a_conversation_rather_than_guessing(self):
        self.bridge.route("U_ME", "hello?", thread_ts="root_nobody_knows")
        self.assertEqual(self.ran[0]["thread_id"], "t_new")

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
        self.assertIn("panel", self.client.text(),
                      "it should say the turn came from somewhere else")
        # And it opened a thread for the conversation it belongs to, rather than
        # dropping the answer loose in the DM.
        self.assertTrue(self.client.posted[0]["text"].startswith("🧵"))
        self.assertTrue(any(m.get("thread_ts") for m in self.client.posted))

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

    # ── a picture or a clip sent from a phone ─────────────────────────────────
    def _file_dm(self, name, text="make this warmer"):
        """The whole inbound path: a Slack file_share event → an input path."""
        import tempfile
        from src.utils.slack_bridge import _route_message, is_actionable

        class Resp:
            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=None):
                return iter([b"MEDIA"])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        event = {"type": "message", "subtype": "file_share", "channel_type": "im",
                 "user": "U_ME", "channel": "D_ME", "text": text,
                 "files": [{"name": name, "url_private_download": "https://x/f"}]}
        self.assertTrue(is_actionable(event),
                        "the event was rejected before anything could download it")
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("requests.get", return_value=Resp()):
                _route_message(self.bridge, mock.Mock(token="t"), event, d)
            return dict(self.ran[0]) if self.ran else {}

    def test_an_image_sent_from_slack_becomes_an_input(self):
        ran = self._file_dm("ref.png")
        self.assertEqual(ran.get("message"), "make this warmer")
        self.assertEqual(len(ran.get("images") or []), 1)
        self.assertTrue(ran["images"][0].endswith("ref.png"))

    def test_a_video_sent_from_slack_becomes_an_input(self):
        ran = self._file_dm("clip.mp4", text="cut this to 5 seconds")
        self.assertTrue(ran["images"][0].endswith("clip.mp4"),
                        "video rides the same path; the pipeline lists it as an input")

    def test_a_bare_attachment_with_no_words_still_runs(self):
        """Sending a photo and nothing else is a complete message to a person."""
        ran = self._file_dm("ref.png", text="")
        self.assertEqual(len(ran.get("images") or []), 1)

    def test_an_attachment_that_did_not_arrive_is_said_out_loud(self):
        """From a phone there is no canvas to look at and no terminal to check,
        so an attachment that quietly did not make it is indistinguishable from
        an agent that ignored it."""
        import logging
        import tempfile
        from src.utils.slack_bridge import _route_message
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        event = {"type": "message", "subtype": "file_share", "channel_type": "im",
                 "user": "U_ME", "channel": "D_ME", "text": "this one",
                 "files": [{"name": "huge.mov", "size": 900 * 1024 * 1024,
                            "url_private_download": "https://x/f"}]}
        with tempfile.TemporaryDirectory() as d:
            _route_message(self.bridge, mock.Mock(token="t"), event, d)
        self.bridge.flush()
        said = self.client.text()
        self.assertIn("Could not take", said)
        self.assertIn("huge.mov", said)

    def test_a_reply_in_the_running_conversation_steers_it(self):
        turn = turn_bus.Turn(request_id="rid_live", thread_id="t_live")
        turn_bus._active["rid_live"] = turn
        self.bridge.on_turn_event({"type": "text", "data": "…"}, turn)
        self.bound[("D_ME", "root_live")] = "t_live"
        with mock.patch("src.utils.interject_bus.post", return_value=True) as post:
            out = self.bridge.route("U_ME", "actually, make it warmer",
                                    thread_ts="root_live")
        self.assertEqual(out["action"], "interject")
        post.assert_called_once()
        self.assertEqual(self.ran, [], "it must not have started a second turn")

    def test_a_message_elsewhere_during_a_run_waits_instead(self):
        """One pipeline. Putting it into the running chat would be the wrong
        conversation; running it alongside would corrupt both."""
        turn = turn_bus.Turn(request_id="rid_live", thread_id="t_live")
        turn_bus._active["rid_live"] = turn
        self.bridge.on_turn_event({"type": "text", "data": "…"}, turn)
        out = self.bridge.route("U_ME", "something else entirely")
        self.assertEqual(out["action"], "busy")
        self.assertEqual(self.ran, [])


if __name__ == "__main__":
    unittest.main()
