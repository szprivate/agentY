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

import os
import unittest
from unittest import mock

from src.utils import turn_bus
from src.utils.slack_bridge import (
    _CONNECT_HINTS, SlackBridge, _why, download_files, is_actionable,
    token_complaint)


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


class TokenTest(unittest.TestCase):
    """Slack issues two kinds of token and they are not interchangeable.

    From a real setup: both fields held the same `xoxb-` bot token, and the only
    symptom was `apps.connections.open` answering `not_allowed_token_type` at the
    bottom of an SDK stack trace — which names neither the field that is wrong
    nor the fact that the other token has to be created separately.
    """

    def _env(self, bot, app):
        keep = {k: os.environ.get(k) for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN")}

        def restore():
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)
        for k, v in (("SLACK_BOT_TOKEN", bot), ("SLACK_APP_TOKEN", app)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_a_matching_pair_has_nothing_to_say(self):
        self._env("xoxb-real", "xapp-real")
        self.assertEqual(token_complaint(), "")

    def test_the_bot_token_pasted_into_both_fields_is_named_exactly(self):
        self._env("xoxb-real", "xoxb-real")
        got = token_complaint()
        self.assertIn("SLACK_APP_TOKEN", got)
        self.assertIn("xoxb-", got)

    def test_swapped_tokens_are_told_apart(self):
        self._env("xapp-real", "xoxb-real")
        self.assertIn("SLACK_BOT_TOKEN", token_complaint())

    def test_something_that_is_neither_says_where_to_get_one(self):
        self._env("xoxb-real", "hunter2")
        got = token_complaint()
        self.assertIn("App-Level Tokens", got)
        self.assertIn("connections:write", got)

    def test_a_missing_token_is_reported_as_missing(self):
        self._env("xoxb-real", None)
        self.assertIn("not set", token_complaint())

    def test_slack_s_own_error_code_is_what_gets_reported(self):
        """It is the searchable half of any failure here."""
        class Boom(Exception):
            response = {"error": "not_allowed_token_type"}
        self.assertEqual(_why(Boom()), "not_allowed_token_type")

    def test_an_error_with_no_code_still_says_something(self):
        self.assertIn("RuntimeError", _why(RuntimeError("socket closed")))

    def test_the_code_that_started_this_has_an_answer_written_for_it(self):
        hint = _CONNECT_HINTS["not_allowed_token_type"]
        self.assertIn("xapp-", hint)
        self.assertIn("App-Level Tokens", hint)

    def test_start_refuses_BEFORE_it_dials_slack(self):
        """The diagnostic is worth nothing if start() does not consult it: the
        SDK would go ahead and fail with the stack trace this exists to replace.
        """
        from src.utils import slack_bridge as sb
        self._env("xoxb-real", "xoxb-real")
        said = []
        with mock.patch.object(sb, "enabled", return_value=True), \
             mock.patch.object(sb, "_complain", said.append):
            started = sb.start(start_turn=None, answer=None, interject=None)
        self.assertFalse(started)
        self.assertTrue(said, "it dialled Slack without complaining first")
        self.assertIn("SLACK_APP_TOKEN", said[0])


class SendFilesTest(unittest.TestCase):
    """The agent handing a file over on purpose.

    Distinct from the mirror, which uploads what a run produced. This is for the
    rest: the JSON it wrote, one frame picked out of sixty, a log worth reading.
    Every file is checked here rather than in the worker so the agent learns what
    it actually sent while it is still writing the sentence about it.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.b = _bridge()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.p = Path(self.dir.name)

    def _file(self, name, size=8):
        f = self.p / name
        f.write_bytes(b"x" * size)
        return str(f)

    def test_a_file_is_uploaded(self):
        out = self.b.send_files([self._file("notes.json")])
        self.b.flush()
        self.assertEqual(len(out["sent"]), 1)
        self.assertEqual(self.b.client.uploaded[0]["filename"], "notes.json")

    def test_any_kind_of_file_goes(self):
        """Not just media — a JSON or a script is often the useful thing."""
        names = ["a.png", "b.mp4", "c.wav", "d.json", "e.py", "f.log"]
        self.b.send_files([self._file(n) for n in names])
        self.b.flush()
        self.assertEqual([u["filename"] for u in self.b.client.uploaded], names)

    def test_the_message_is_posted_above_them(self):
        self.b.send_files([self._file("a.png")], message="**here** is the frame")
        self.b.flush()
        self.assertIn("*here* is the frame", self.b.client.posted[0]["text"],
                      "and converted to Slack's markdown")

    def test_no_message_posts_no_message(self):
        self.b.send_files([self._file("a.png")])
        self.b.flush()
        self.assertEqual(self.b.client.posted, [])

    def test_a_path_that_is_not_there_is_reported_not_silently_skipped(self):
        out = self.b.send_files([self._file("real.png"), "D:/nope/ghost.png"])
        self.assertEqual(len(out["sent"]), 1)
        self.assertEqual(out["missing"], ["D:/nope/ghost.png"])

    def test_one_bad_path_does_not_lose_the_good_ones(self):
        self.b.send_files(["D:/nope/ghost.png", self._file("real.png")])
        self.b.flush()
        self.assertEqual(len(self.b.client.uploaded), 1)

    def test_a_file_slack_would_reject_is_named_rather_than_attempted(self):
        with mock.patch("src.utils.slack_bridge._setting",
                        side_effect=lambda k, d: 0 if k == "max_upload_mb" else d):
            out = self.b.send_files([self._file("huge.mp4", size=4096)])
        self.b.flush()
        self.assertEqual(out["sent"], [])
        self.assertEqual(len(out["too_large"]), 1)
        self.assertEqual(self.b.client.uploaded, [])

    def test_with_nowhere_to_post_it_says_so(self):
        b = _bridge(default_channel="")
        self.assertIn("nowhere to post", b.send_files([self._file("a.png")])["error"])

    def test_uploading_happens_off_the_calling_thread(self):
        self.b.send_files([self._file("a.png")])
        self.assertEqual(self.b.client.uploaded, [], "it uploaded inline")
        self.b.flush()
        self.assertTrue(self.b.client.uploaded)


class InboundEventTest(unittest.TestCase):
    """Slack sends a great deal that looks like a message and is not."""

    def _ev(self, **kw):
        base = {"type": "message", "channel_type": "im", "user": "U_ME", "text": "hi"}
        base.update(kw)
        return base

    def test_a_plain_dm_is_actionable(self):
        self.assertTrue(is_actionable(self._ev()))

    def test_a_dm_with_an_attachment_is_actionable(self):
        """THE bug this rule had: to Slack, sending a picture is not a different
        kind of event — it is a message with `subtype: file_share`. Rejecting
        every subtype drops exactly the message somebody took a photo for, and
        drops it in silence.
        """
        self.assertTrue(is_actionable(self._ev(subtype="file_share", files=[{}])))

    def test_an_attachment_with_no_words_is_still_actionable(self):
        self.assertTrue(is_actionable(self._ev(subtype="file_share", text="", files=[{}])))

    def test_an_edit_is_not(self):
        """Editing an old message must not re-run it."""
        self.assertFalse(is_actionable(self._ev(subtype="message_changed")))

    def test_a_deletion_is_not(self):
        self.assertFalse(is_actionable(self._ev(subtype="message_deleted")))

    def test_some_other_subtype_is_still_rejected(self):
        """`file_share` is an exception, not the end of the rule."""
        for sub in ("channel_join", "thread_broadcast", "bot_message", "me_message"):
            self.assertFalse(is_actionable(self._ev(subtype=sub)), sub)

    def test_a_bot_post_is_not(self):
        self.assertFalse(is_actionable(self._ev(bot_id="B1")))

    def test_a_channel_message_is_not(self):
        self.assertFalse(is_actionable(self._ev(channel_type="channel")))

    def test_something_that_is_not_a_message_is_not(self):
        self.assertFalse(is_actionable(self._ev(type="reaction_added")))

    def test_a_message_from_nobody_is_not(self):
        self.assertFalse(is_actionable(self._ev(user="")))


class _Resp:
    """A streaming requests response, which is what a phone video needs."""

    def __init__(self, chunks=(b"PNGDATA",)):
        self._chunks = list(chunks)

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class AttachmentTest(unittest.TestCase):
    """What someone sends the agent from a phone."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _ev(self, *files):
        return {"files": [dict({"url_private_download": "https://x/f"}, **f)
                          for f in files]}

    def _get(self, resp=None, **kw):
        return mock.patch("requests.get", return_value=resp or _Resp(), **kw)

    def _hush(self):
        import logging
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_an_attached_image_is_saved_and_handed_over_as_an_input(self):
        from pathlib import Path
        with self._get() as get:
            paths, skipped = download_files(mock.Mock(token="xoxb-1"),
                                            self._ev({"name": "ref.png"}), self.dir.name)
        self.assertEqual(len(paths), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(Path(paths[0]).read_bytes(), b"PNGDATA")
        self.assertIn("Bearer xoxb-1", get.call_args.kwargs["headers"]["Authorization"])

    def test_a_video_arrives_the_same_way(self):
        """The pipeline lists video paths as inputs; nothing here is image-only."""
        paths, _ = self._download({"name": "clip.mp4", "size": 5_000_000})
        self.assertTrue(paths[0].endswith("clip.mp4"))

    def _download(self, *files, resp=None):
        with self._get(resp):
            return download_files(mock.Mock(token="t"), self._ev(*files), self.dir.name)

    def test_it_streams_rather_than_holding_the_whole_file(self):
        """A video filmed on a phone is measured in hundreds of megabytes."""
        from pathlib import Path
        paths, _ = self._download({"name": "big.mp4"},
                                  resp=_Resp([b"a" * 1024, b"b" * 1024]))
        self.assertEqual(Path(paths[0]).stat().st_size, 2048)

    def test_a_file_over_the_limit_is_refused_by_its_declared_size(self):
        paths, skipped = self._download({"name": "huge.mov", "size": 999 * 1024 * 1024})
        self.assertEqual(paths, [])
        self.assertIn("huge.mov", skipped[0])
        self.assertIn("MB limit", skipped[0])

    def test_a_file_that_lies_about_its_size_is_stopped_mid_download(self):
        """`size` is Slack's word for it; the bytes are the fact.

        No declared size at all here, so the pre-check cannot fire and only the
        running total can stop it.
        """
        self._hush()
        with mock.patch("src.utils.slack_bridge._setting",
                        side_effect=lambda k, d: 0 if k == "max_download_mb" else d):
            paths, skipped = self._download({"name": "liar.mp4"},
                                            resp=_Resp([b"x" * 4096]))
        self.assertEqual(paths, [])
        self.assertTrue(skipped)

    def test_half_a_video_is_not_left_on_disk(self):
        from pathlib import Path
        self._hush()
        with mock.patch("src.utils.slack_bridge._setting",
                        side_effect=lambda k, d: 0 if k == "max_download_mb" else d):
            self._download({"name": "liar.mp4"}, resp=_Resp([b"x" * 4096]))
        self.assertEqual(list(Path(self.dir.name).glob("*")), [])

    def test_a_download_that_fails_does_not_lose_the_message(self):
        self._hush()
        with mock.patch("requests.get", side_effect=RuntimeError("no")):
            paths, skipped = download_files(mock.Mock(token="t"),
                                            self._ev({"name": "x.png"}), self.dir.name)
        self.assertEqual(paths, [])
        self.assertIn("x.png", skipped[0])

    def test_one_bad_attachment_does_not_lose_the_good_one(self):
        self._hush()
        calls = [RuntimeError("no"), _Resp()]
        with mock.patch("requests.get", side_effect=calls):
            paths, skipped = download_files(
                mock.Mock(token="t"),
                self._ev({"name": "bad.png"}, {"name": "good.png"}), self.dir.name)
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(skipped), 1)

    def test_a_file_with_no_link_names_the_scope_that_is_missing(self):
        paths, skipped = download_files(
            mock.Mock(token="t"),
            {"files": [{"name": "x.png"}]}, self.dir.name)
        self.assertEqual(paths, [])
        self.assertIn("files:read", skipped[0])

    def test_more_attachments_than_a_turn_can_use_are_capped(self):
        from src.utils.slack_bridge import _MAX_INBOUND_FILES
        files = [{"name": f"{i}.png"} for i in range(_MAX_INBOUND_FILES + 3)]
        paths, skipped = self._download(*files)
        self.assertEqual(len(paths), _MAX_INBOUND_FILES)
        self.assertIn("only the first", skipped[0])

    def test_the_sender_s_filename_cannot_escape_the_download_folder(self):
        from pathlib import Path
        paths, _ = self._download({"name": "../../etc/passwd"})
        self.assertEqual(Path(paths[0]).parent, Path(self.dir.name))


if __name__ == "__main__":
    unittest.main()
