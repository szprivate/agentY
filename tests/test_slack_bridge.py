"""Slack as a second line in: who may drive it, and what a message means.

Three things a DM can be, and the order they are resolved in matters more than
any of them individually:

1. the **answer** to a question the agent is holding a turn open for;
2. an **interjection** into a turn that is already running — because starting a
   second turn would drive two of them through one pipeline singleton, and both
   would be wrong;
3. otherwise, a **new turn**.

Plus the part that is not a convenience: nobody outside the allow-list gets any
of the three. A DM to this bot runs generations, tools and scripts on someone's
workstation, and "whoever found the app" is not an access rule.

    python -m unittest discover -s tests
"""

import queue
import unittest
from unittest import mock

from src.utils import turn_bus
from src.utils.slack_bridge import SlackBridge, download_files, is_actionable


class FakeClient:
    """Records what would have been said, in order."""

    def __init__(self):
        self.posted = []
        self.updated = []
        self.deleted = []
        self.uploaded = []
        self._ts = 0

    def chat_postMessage(self, **kw):
        self._ts += 1
        self.posted.append(kw)
        return {"ts": f"ts{self._ts}"}

    def chat_update(self, **kw):
        self.updated.append(kw)
        return {"ok": True}

    def chat_delete(self, **kw):
        self.deleted.append(kw)
        return {"ok": True}

    def files_upload_v2(self, **kw):
        self.uploaded.append(kw)
        return {"ok": True}


def _bridge(**kw):
    kw.setdefault("client", FakeClient())
    kw.setdefault("allowed_users", ["U_ME"])
    kw.setdefault("default_channel", "D_ME")
    b = SlackBridge(**kw)
    b.bot_user_id = "U_BOT"
    return b


class WhoMayTalkTest(unittest.TestCase):

    def test_the_owner_may(self):
        b = _bridge(start_turn=lambda t, f: "r1")
        self.assertEqual(b.route("U_ME", "render it")["action"], "turn")

    def test_a_stranger_may_not(self):
        b = _bridge(start_turn=lambda t, f: "r1")
        out = b.route("U_SOMEONE", "render it")
        self.assertEqual(out["action"], "denied")

    def test_an_empty_allow_list_refuses_everyone(self):
        """Not a default worth having: it would hand the machine to the workspace."""
        import logging
        logging.disable(logging.CRITICAL)   # the refusal warns, on purpose
        self.addCleanup(logging.disable, logging.NOTSET)
        b = _bridge(allowed_users=[], start_turn=lambda t, f: "r1")
        out = b.route("U_ME", "render it")
        self.assertEqual(out["action"], "denied")
        # The reason matters as much as the refusal: "not configured" is the one
        # that tells the owner why their own messages are being ignored. Landing
        # in the ordinary "not on the list" branch would refuse identically and
        # explain nothing.
        self.assertEqual(out["why"], "no allow-list configured")

    def test_the_bot_does_not_answer_itself(self):
        """Every mirrored message is posted BY the bot — without this it loops."""
        b = _bridge(start_turn=lambda t, f: "r1")
        self.assertEqual(b.route("U_BOT", "🔧 tool")["action"], "ignored")

    def test_an_empty_message_is_not_a_turn(self):
        b = _bridge(start_turn=lambda t, f: "r1")
        self.assertEqual(b.route("U_ME", "   ")["action"], "ignored")

    def test_a_bare_image_is_a_message(self):
        """A picture with no words still means 'do something with this'."""
        seen = {}
        b = _bridge(start_turn=lambda t, f: seen.update(text=t, files=f) or "r1")
        self.assertEqual(b.route("U_ME", "", ["C:/a.png"])["action"], "turn")
        self.assertEqual(seen["files"], ["C:/a.png"])


class WhatAMessageMeansTest(unittest.TestCase):

    def setUp(self):
        self.started, self.answered, self.interjected = [], [], []
        self.b = _bridge(
            start_turn=lambda t, f: self.started.append(t) or "r_new",
            answer=lambda rid, t: self.answered.append((rid, t)) or True,
            interject=lambda rid, t: self.interjected.append((rid, t)) or True)
        self.addCleanup(turn_bus._active.clear)

    def _turn(self, rid="r1", *, asking=False, ended=False):
        from src.utils.slack_bridge import SlackTurn
        st = SlackTurn(self.b, turn_bus.Turn(request_id=rid, thread_id="t"), "D_ME")
        if asking:
            st.ask_request_id = rid
        if ended:
            st.ended = 1.0
        self.b.turns[rid] = st
        return st

    def _running(self, rid="r1"):
        turn_bus._active[rid] = turn_bus.Turn(request_id=rid, thread_id="t")

    def test_a_pending_question_takes_the_message_as_its_answer(self):
        self._turn(asking=True)
        self._running()
        self.assertEqual(self.b.route("U_ME", "yes")["action"], "answer")
        self.assertEqual(self.answered, [("r1", "yes")])

    def test_answering_beats_interjecting(self):
        """The agent is *waiting*; an interjection would leave it waiting."""
        self._turn(asking=True)
        self._running()
        self.b.route("U_ME", "yes")
        self.assertEqual(self.interjected, [])

    def test_a_running_turn_is_interjected_not_restarted(self):
        """Two turns through one pipeline singleton corrupts both."""
        self._turn()
        self._running()
        self.assertEqual(self.b.route("U_ME", "actually, warmer")["action"], "interject")
        self.assertEqual(self.started, [])

    def test_with_nothing_running_it_starts_a_turn(self):
        self.assertEqual(self.b.route("U_ME", "render it")["action"], "turn")
        self.assertEqual(self.started, ["render it"])

    def test_a_finished_turn_does_not_swallow_the_next_message(self):
        self._turn(ended=True)
        self.assertEqual(self.b.route("U_ME", "again please")["action"], "turn")

    def test_an_ask_that_is_no_longer_pending_falls_through(self):
        """The panel answered it first — the reply registry says so."""
        self.b._answer = lambda rid, t: False
        self._turn(asking=True)
        self.assertEqual(self.b.route("U_ME", "yes")["action"], "turn")


class MirrorTest(unittest.TestCase):
    """What actually reaches Slack while a turn runs."""

    def setUp(self):
        self.client = FakeClient()
        self.b = _bridge(client=self.client)
        self.turn = turn_bus.Turn(request_id="r1", thread_id="t1",
                                  origin="panel", text="make a hero sheet")

    def _feed(self, *events):
        for ev in events:
            self.b.on_turn_event(ev, self.turn)
        # What the worker thread does continuously: land the throttled edit, then
        # make the queued Slack calls.
        self.b._tick_turns()
        self.b.flush()

    def test_a_panel_turn_shows_up_in_slack(self):
        """The whole point: work started at the desk, watched from a phone."""
        self._feed({"type": "text", "data": "On it."})
        self.assertTrue(self.client.posted, "nothing was posted at all")

    def test_the_answer_is_one_message_that_gets_rewritten(self):
        self._feed({"type": "text", "data": "Rendered "},
                   {"type": "text", "data": "four frames."},
                   {"type": "done"})
        self.assertEqual(len(self.client.posted), 1, "one message per turn")
        self.assertIn("four frames.", self.client.updated[-1]["text"])

    def test_detail_goes_into_that_message_s_thread(self):
        self._feed({"type": "text", "data": "hi"},
                   {"type": "tool", "phase": "call", "id": "t1", "name": "run_research"})
        threaded = [p for p in self.client.posted if p.get("thread_ts")]
        self.assertTrue(threaded, "the tool call was not threaded")
        self.assertIn("run_research", threaded[0]["text"])

    def test_a_file_is_uploaded_to_the_channel(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hero.png"
            p.write_bytes(b"x")
            self._feed({"type": "output", "path": str(p), "name": "hero.png"})
            self.b.flush()
        self.assertEqual(self.client.uploaded[0]["filename"], "hero.png")

    def test_a_file_that_vanished_is_reported_rather_than_dropped(self):
        self._feed({"type": "output", "path": "D:/gone/none.png", "name": "none.png"})
        self.b.flush()
        self.assertIn("not on disk", self.client.posted[-1]["text"])

    def test_the_status_line_is_rewritten_not_repeated(self):
        self._feed({"type": "text", "data": "hi"},
                   {"type": "progress", "data": "step 1"},
                   {"type": "progress", "data": "step 2"})
        self.b.flush()
        self.assertEqual(len([p for p in self.client.posted
                              if "step" in p.get("text", "")]), 1)
        self.assertIn("step 2", self.client.updated[-1]["text"])

    def test_a_turn_that_produced_nothing_visible_posts_nothing(self):
        """A turn nobody watched should not leave an empty stub in the DM."""
        self.b.on_turn_event({"type": "done"}, self.turn)
        self.assertEqual(self.client.posted, [])

    def test_with_no_channel_it_stays_quiet(self):
        b = _bridge(client=FakeClient(), default_channel="")
        b.on_turn_event({"type": "text", "data": "hi"}, self.turn)
        self.assertEqual(b.client.posted, [])

    def test_a_slack_api_failure_does_not_escape_into_the_turn(self):
        class Broken(FakeClient):
            def chat_postMessage(self, **kw):
                raise RuntimeError("slack is down")
        b = _bridge(client=Broken())
        import logging
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        b.on_turn_event({"type": "text", "data": "hi"}, self.turn)   # must not raise
        b.flush()                                                    # nor here
        b._tick_turns()

    def test_slack_calls_happen_off_the_turn_s_thread(self):
        """A hook run makes dozens of them; a round trip between the agent and
        its next step would make the PANEL slow because a phone was watching."""
        self.b.on_turn_event({"type": "text", "data": "hi"}, self.turn)
        self.b.on_turn_event({"type": "tool", "phase": "call", "id": "t",
                              "name": "go"}, self.turn)
        self.assertEqual(self.client.posted, [], "it talked to Slack inline")
        self.b.flush()
        self.assertTrue(self.client.posted)


class InboundEventTest(unittest.TestCase):
    """Slack sends a great deal that looks like a message and is not."""

    def _ev(self, **kw):
        base = {"type": "message", "channel_type": "im", "user": "U_ME", "text": "hi"}
        base.update(kw)
        return base

    def test_a_plain_dm_is_actionable(self):
        self.assertTrue(is_actionable(self._ev()))

    def test_an_edit_is_not(self):
        """Editing an old message must not re-run it."""
        self.assertFalse(is_actionable(self._ev(subtype="message_changed")))

    def test_a_bot_post_is_not(self):
        self.assertFalse(is_actionable(self._ev(bot_id="B1")))

    def test_a_channel_message_is_not(self):
        self.assertFalse(is_actionable(self._ev(channel_type="channel")))

    def test_something_that_is_not_a_message_is_not(self):
        self.assertFalse(is_actionable(self._ev(type="reaction_added")))

    def test_a_message_from_nobody_is_not(self):
        self.assertFalse(is_actionable(self._ev(user="")))


class AttachmentTest(unittest.TestCase):

    def test_an_attached_image_is_saved_and_handed_over_as_an_input(self):
        import tempfile
        from pathlib import Path

        class Resp:
            content = b"PNGDATA"
            def raise_for_status(self):
                return None

        ev = {"files": [{"name": "ref.png", "url_private_download": "https://x/ref.png"}]}
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("requests.get", return_value=Resp()) as get:
                paths = download_files(mock.Mock(token="xoxb-1"), ev, d)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).read_bytes(), b"PNGDATA")
            self.assertIn("Bearer xoxb-1", get.call_args.kwargs["headers"]["Authorization"])

    def test_a_download_that_fails_does_not_lose_the_message(self):
        import logging
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        ev = {"files": [{"name": "x.png", "url_private_download": "https://x/x.png"}]}
        with mock.patch("requests.get", side_effect=RuntimeError("no")):
            self.assertEqual(download_files(mock.Mock(token="t"), ev, "."), [])


if __name__ == "__main__":
    unittest.main()
