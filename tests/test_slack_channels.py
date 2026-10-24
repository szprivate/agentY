"""Slack channels with several people: @mention only, a thread per request, and
the person who asked is the one who answers for it.

The DM bridge is covered by test_slack_bridge.py / test_slack_threads.py /
test_slack_seam.py and must keep passing unchanged; this file is only about what
a channel adds. Slack and the pipeline are replaced; the bridge, the queue and
the team store are real.

    python -m unittest tests.test_slack_channels
"""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from src.utils import team_store, turn_bus, turn_queue
from src.utils.slack_bridge import (
    SlackBridge, _Approval, _route_message, is_actionable, parse_decision)


class FakeClient:
    def __init__(self):
        self.posted, self.updated, self.deleted, self.uploaded = [], [], [], []
        self.reactions = []
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

    def reactions_add(self, **kw):
        self.reactions.append(("+", kw["name"]))
        return {"ok": True}

    def reactions_remove(self, **kw):
        self.reactions.append(("-", kw["name"]))
        return {"ok": True}

    def users_info(self, user):
        return {"user": {"real_name": "Anna " + user}}


class _Store:
    def __init__(self):
        self.bound = {}

    def set(self, tid, ch, ts):
        self.bound[(ch, ts)] = tid

    def get(self, tid):
        return next(({"channel": c, "root_ts": s}
                     for (c, s), v in self.bound.items() if v == tid), None)

    def reverse(self, ch, ts):
        return self.bound.get((ch, ts))

    def list(self, limit=200):
        return []


class ChannelCase(unittest.TestCase):
    """A bridge with one allowed channel, a real queue and a recording starter."""

    LIMIT = 1

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.enterContext(mock.patch.dict(os.environ, {"AGENTY_TEAM_DIR": tmp.name}))
        team_store._requesters.clear()
        team_store._loaded = False
        self.addCleanup(team_store._requesters.clear)
        self.store = _Store()
        for name, fn in (("set_slack_thread", self.store.set),
                         ("get_slack_thread", self.store.get),
                         ("thread_for_slack", self.store.reverse),
                         ("list_threads", self.store.list)):
            self.enterContext(mock.patch("src.utils.slack_bridge.cs." + name, fn))
        self.client = FakeClient()
        self.started = []
        self.queue = turn_queue.TurnQueue(limit=self.LIMIT, max_queued=3)
        self.b = SlackBridge(
            client=self.client, allowed_users=["U_ANNA", "U_BEN"], default_channel="D_ME",
            allowed_channels=["C_TEAM"], approvers=["U_BOSS"], turns_queue=self.queue,
            start_turn=self._start, answer=self._answer, interject=lambda r, t: True,
            members_may_ask=False, queue_dm=False)
        self.b.bot_user_id = "U_BOT"
        self.queue.add_listener(self.b.on_queue_change)
        self.answers = []
        self.addCleanup(turn_bus._active.clear)

    def _start(self, text, files, thread_id="", **extra):
        rid = f"rid{len(self.started) + 1}"
        tid = thread_id or f"conv{len(self.started) + 1}"
        root = extra.get("slack_root")
        if root:
            self.store.set(tid, *root)
        if extra.get("requester"):
            team_store.set_requester(tid, **extra["requester"])
        self.started.append({"text": text, "tid": tid, "rid": rid, "extra": extra})
        return rid

    def _answer(self, rid, text):
        self.answers.append((rid, text))
        return True

    def mention(self, user, text, ts="100.1", thread_ts="", **kw):
        return self.b.route(user, text, channel="C_TEAM", channel_type="channel",
                            mentioned=True, msg_ts=ts, thread_ts=thread_ts,
                            user_name="Name " + user, **kw)

    def posts(self):
        self.b.flush()
        return [p.get("text", "") for p in self.client.posted]


class MentionRulesTest(ChannelCase):

    def test_a_mention_in_an_allowed_channel_starts_a_turn_in_its_own_thread(self):
        out = self.mention("U_ANNA", "<@U_BOT> make a poster")
        self.assertEqual(out["action"], "turn")
        run = self.started[0]
        self.assertEqual(run["text"], "make a poster")           # the mention is stripped
        self.assertEqual(run["extra"]["slack_root"], ("C_TEAM", "100.1"))
        self.assertEqual(run["extra"]["requester"]["user"], "U_ANNA")
        self.assertEqual(run["extra"]["requester"]["project"], "C_TEAM")

    def test_the_bound_thread_is_the_requesters_own_message(self):
        self.mention("U_ANNA", "make a poster")
        self.assertEqual(self.store.reverse("C_TEAM", "100.1"), "conv1")

    def test_a_message_without_a_mention_is_ignored(self):
        out = self.b.route("U_ANNA", "lunch?", channel="C_TEAM", channel_type="channel",
                           mentioned=False, msg_ts="1.1")
        self.assertEqual(out["action"], "ignored")
        self.assertEqual(self.started, [])

    def test_a_channel_that_is_not_listed_is_ignored(self):
        out = self.b.route("U_ANNA", "hi", channel="C_OTHER", channel_type="channel",
                           mentioned=True, msg_ts="1.1")
        self.assertEqual(out["action"], "ignored")

    def test_a_stranger_in_an_allowed_channel_is_refused(self):
        out = self.mention("U_STRANGER", "run script")
        self.assertEqual(out["action"], "denied")
        self.assertEqual(self.started, [])

    def test_members_may_ask_lets_the_channel_in(self):
        self.b.members_may_ask = True
        self.assertEqual(self.mention("U_STRANGER", "a cat")["action"], "turn")

    def test_an_empty_allow_list_still_refuses_in_a_channel(self):
        self.b.allowed = []
        self.assertEqual(self.mention("U_ANNA", "a cat")["action"], "denied")

    def test_a_reply_in_the_thread_continues_that_conversation(self):
        self.mention("U_ANNA", "make a poster", ts="100.1")
        # the first turn is still running (limit 1); a reply steers it
        with mock.patch.object(turn_bus, "active",
                               return_value=[turn_bus.Turn("rid1", "conv1")]):
            out = self.mention("U_ANNA", "bluer please", ts="100.2", thread_ts="100.1")
        self.assertEqual(out["action"], "interject")
        self.assertEqual(len(self.started), 1)


class WhoMayAnswerTest(ChannelCase):

    def setUp(self):
        super().setUp()
        self.mention("U_ANNA", "install that node pack", ts="200.1")

    def test_the_requester_may_steer(self):
        self.assertTrue(self.b.may_steer("U_ANNA", "conv1"))

    def test_an_approver_may_steer(self):
        self.assertTrue(self.b.may_steer("U_BOSS", "conv1"))

    def test_a_bystander_may_not(self):
        self.assertFalse(self.b.may_steer("U_BEN", "conv1"))

    def test_a_user_with_the_approver_role_may(self):
        with mock.patch("src.utils.user_auth.slack_roles", return_value={"U_BEN": ["approver"]}):
            self.assertTrue(self.b.may_steer("U_BEN", "conv1"))

    def test_a_conversation_nobody_owns_answers_to_the_allow_list(self):
        self.assertTrue(self.b.may_steer("U_BEN", "someone-elses-dm-conversation"))

    def test_a_bystander_cannot_answer_an_ask(self):
        from src.utils.slack_bridge import SlackTurn
        turn = turn_bus.Turn("rid1", "conv1")
        st = SlackTurn(self.b, turn, "C_TEAM", root_ts="200.1")
        st.ask_request_id = "ask1"
        self.b.turns["rid1"] = st
        out = self.mention("U_BEN", "yes do it", ts="200.2", thread_ts="200.1")
        self.assertEqual(out["action"], "denied")
        self.assertEqual(self.answers, [])
        out = self.mention("U_ANNA", "yes", ts="200.3", thread_ts="200.1")
        self.assertEqual(out["action"], "answer")
        self.assertEqual(self.answers, [("ask1", "yes")])

    def test_a_bystander_cannot_interject_into_the_turn(self):
        with mock.patch.object(turn_bus, "active",
                               return_value=[turn_bus.Turn("rid1", "conv1")]):
            out = self.mention("U_BEN", "actually, use red", ts="200.2", thread_ts="200.1")
        self.assertEqual(out["action"], "denied")


class ApprovalInThreadTest(ChannelCase):
    """A tool that needs a yes asks in the thread, and only the right people count."""

    def setUp(self):
        super().setUp()
        self.mention("U_ANNA", "install the node", ts="300.1")

    def _ask(self, results, timeout=5.0):
        results.append(self.b.ask_approval("conv1", "install_custom_node",
                                           "git clone https://example/x", timeout))

    def _wait_pending(self):
        for _ in range(200):
            if self.b._pending_approval("conv1") is not None:
                return
            threading.Event().wait(0.01)
        self.fail("the approval was never asked")

    def test_the_card_names_the_requester_and_goes_in_the_thread(self):
        res = []
        t = threading.Thread(target=self._ask, args=(res,))
        t.start()
        self._wait_pending()
        card = self.client.posted[-1]
        self.assertEqual(card["thread_ts"], "300.1")
        self.assertEqual(card["channel"], "C_TEAM")
        self.assertIn("<@U_ANNA>", card["text"])
        self.assertIn("install_custom_node", card["text"])
        self.b.route("U_ANNA", "yes", channel="C_TEAM", channel_type="channel",
                     mentioned=False, msg_ts="300.2", thread_ts="300.1")
        t.join(5)
        self.assertTrue(res[0]["allowed"])

    def test_a_no_declines(self):
        res = []
        t = threading.Thread(target=self._ask, args=(res,))
        t.start()
        self._wait_pending()
        self.b.route("U_BOSS", "no", channel="C_TEAM", channel_type="channel",
                     mentioned=True, msg_ts="300.2", thread_ts="300.1")
        t.join(5)
        self.assertFalse(res[0]["allowed"])
        self.assertIn("U_BOSS", res[0]["note"])

    def test_a_bystanders_yes_does_not_approve(self):
        res = []
        t = threading.Thread(target=self._ask, args=(res, 0.4))
        t.start()
        self._wait_pending()
        out = self.b.route("U_BEN", "yes", channel="C_TEAM", channel_type="channel",
                           mentioned=True, msg_ts="300.2", thread_ts="300.1")
        self.assertEqual(out["action"], "denied")
        t.join(5)
        self.assertFalse(res[0]["allowed"])          # timed out: nobody valid said yes

    def test_something_that_is_not_yes_or_no_does_not_decide(self):
        res = []
        t = threading.Thread(target=self._ask, args=(res, 0.4))
        t.start()
        self._wait_pending()
        out = self.b.route("U_ANNA", "what does it do?", channel="C_TEAM",
                           channel_type="channel", mentioned=True, msg_ts="300.2",
                           thread_ts="300.1")
        self.assertEqual(out["action"], "approval_unclear")
        self.assertEqual(self.answers, [])           # and it was not sent on as an answer
        t.join(5)
        self.assertFalse(res[0]["allowed"])

    def test_an_unbound_conversation_is_not_asked_in_slack(self):
        self.assertIsNone(self.b.ask_approval("panel-conversation", "run_script", "x", 1))

    def test_unmentioned_yes_in_the_thread_counts(self):
        """Answering a card is a reply, not a new request: no @ needed."""
        res = []
        t = threading.Thread(target=self._ask, args=(res,))
        t.start()
        self._wait_pending()
        out = self.b.route("U_ANNA", "yes", channel="C_TEAM", channel_type="channel",
                           mentioned=False, msg_ts="300.2", thread_ts="300.1")
        t.join(5)
        self.assertEqual(out["action"], "approval")

    def test_tool_permissions_asks_slack_first(self):
        from src.utils import tool_permissions as tp
        seen = {}

        def asker(thread_id, tool, summary, timeout):
            seen.update(thread=thread_id, tool=tool)
            return {"allowed": True, "note": "ok in slack"}

        tp.set_remote_asker(asker)
        self.addCleanup(tp.set_remote_asker, None)
        with mock.patch.object(tp, "_asking_conversation", return_value={"thread_id": "conv1"}):
            d = tp.request("install_custom_node", {"source": "x"}, timeout=1)
        self.assertTrue(d.allowed)
        self.assertEqual(seen, {"thread": "conv1", "tool": "install_custom_node"})

    def test_tool_permissions_falls_through_when_slack_does_not_own_it(self):
        from src.utils import tool_permissions as tp
        tp.set_remote_asker(lambda *a: None)
        self.addCleanup(tp.set_remote_asker, None)
        d = tp.request("run_script", {"command": "ls"}, timeout=1)
        self.assertFalse(d.allowed)                   # no panel: the old refusal
        self.assertIn("no agentY panel", d.reason if hasattr(d, "reason") else d.message)


class QueueInTheThreadTest(ChannelCase):
    LIMIT = 1

    def test_the_second_request_waits_and_is_told_where_it_stands(self):
        first = self.mention("U_ANNA", "first", ts="400.1")
        second = self.mention("U_BEN", "second", ts="400.2")
        self.assertEqual(first["action"], "turn")
        self.assertEqual(second["action"], "queued")
        self.assertEqual(second["position"], 1)
        self.assertEqual(len(self.started), 1)
        note = [p for p in self.posts() if "Queued" in p]
        self.assertEqual(len(note), 1)
        self.assertEqual(self.client.posted[-1]["thread_ts"], "400.2")   # in ITS thread
        self.assertIn("number 1", note[0])

    def test_it_starts_when_the_running_turn_ends_and_the_note_goes(self):
        self.mention("U_ANNA", "first", ts="400.1")
        self.mention("U_BEN", "second", ts="400.2")
        self.b.flush()
        # the first turn's done event frees the place
        self.b.on_turn_event({"type": "done"}, turn_bus.Turn("rid1", "conv1"))
        self.b.flush()
        self.assertEqual([s["text"] for s in self.started], ["first", "second"])
        self.assertEqual(len(self.client.deleted), 1)       # "Queued" message removed

    def test_a_third_waiting_request_sees_its_number(self):
        self.mention("U_ANNA", "first", ts="400.1")
        self.mention("U_BEN", "second", ts="400.2")
        self.mention("U_BOSS", "third", ts="400.3") if False else None
        third = self.b.route("U_ANNA", "third", channel="C_TEAM", channel_type="channel",
                             mentioned=True, msg_ts="400.3")
        self.assertEqual(third["position"], 2)

    def test_a_follow_up_to_a_waiting_request_joins_it(self):
        self.mention("U_ANNA", "first", ts="400.1")
        self.mention("U_BEN", "second", ts="400.2")
        out = self.mention("U_BEN", "and make it blue", ts="400.3", thread_ts="400.2")
        self.assertEqual(out["action"], "amended")
        self.b.on_turn_event({"type": "done"}, turn_bus.Turn("rid1", "conv1"))
        self.assertIn("and make it blue", self.started[1]["text"])
        self.assertEqual(len(self.started), 2)               # no third turn

    def test_a_full_queue_says_so(self):
        for i in range(4):
            self.b.route(f"U_ANNA", f"r{i}", channel="C_TEAM", channel_type="channel",
                         mentioned=True, msg_ts=f"5.{i}")
        out = self.b.route("U_ANNA", "one too many", channel="C_TEAM", channel_type="channel",
                           mentioned=True, msg_ts="5.9")
        self.assertEqual(out["action"], "busy")


class ParallelTest(ChannelCase):
    LIMIT = 2

    def test_two_requests_run_at_once_when_the_limit_allows(self):
        self.assertEqual(self.mention("U_ANNA", "a", ts="6.1")["action"], "turn")
        self.assertEqual(self.mention("U_BEN", "b", ts="6.2")["action"], "turn")
        self.assertEqual(self.mention("U_ANNA", "c", ts="6.3")["action"], "queued")


class ChannelOutputTest(ChannelCase):

    def test_a_turn_speaks_in_its_channel_thread_and_pings_the_requester(self):
        self.mention("U_ANNA", "poster", ts="700.1")
        turn = turn_bus.Turn("rid1", "conv1", origin="slack", text="poster")
        self.b.on_turn_event({"type": "ask", "request_id": "q1",
                              "prompt": "Which size?"}, turn)
        self.b.on_turn_event({"type": "text", "data": "Working on it"}, turn)
        self.b.flush()
        self.b._tick_turns()
        self.b.flush()
        asked = [p for p in self.client.posted if "waiting on you" in p["text"]][0]
        self.assertEqual((asked["channel"], asked["thread_ts"]), ("C_TEAM", "700.1"))
        self.assertTrue(asked["text"].startswith("<@U_ANNA>"))
        self.assertNotIn("reply_broadcast", asked)          # not shouted at the channel
        self.assertTrue(all(p["channel"] == "C_TEAM" for p in self.client.posted))

    def test_files_the_agent_sends_go_to_that_thread_not_the_dm(self):
        self.mention("U_ANNA", "poster", ts="700.1")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "shots.json"
            f.write_text("{}")
            out = self.b.send_files([str(f)], "here you go", thread_id="conv1")
            self.b.flush()
        self.assertEqual(out["sent"], [str(f)])
        self.assertEqual(self.client.uploaded[0]["channel"], "C_TEAM")
        self.assertEqual(self.client.uploaded[0]["thread_ts"], "700.1")

    def test_a_panel_turn_still_mirrors_to_the_dm(self):
        turn = turn_bus.Turn("ridP", "panel-conv", origin="panel", text="hi")
        self.b.on_turn_event({"type": "text", "data": "hello"}, turn)
        self.b.on_turn_event({"type": "done"}, turn)
        self.b.flush()
        self.assertTrue(self.client.posted)
        self.assertTrue(all(p["channel"] == "D_ME" for p in self.client.posted))


class CommandsTest(ChannelCase):

    def test_remember_for_me_and_for_the_channel(self):
        out = self.mention("U_ANNA", "remember: I always want 16:9")
        self.assertEqual(out["action"], "command")
        self.mention("U_ANNA", "remember for project: client is Acme, keep logos small")
        self.assertIn("16:9", team_store.read_notes("users", "U_ANNA"))
        self.assertIn("Acme", team_store.read_notes("projects", "C_TEAM"))
        self.assertEqual(self.started, [])                  # neither was a turn

    def test_notes_are_read_into_the_next_turn_context(self):
        team_store.add_note("users", "U_ANNA", "I always want 16:9")
        team_store.add_note("projects", "C_TEAM", "client is Acme")
        block = team_store.context_block({"user": "U_ANNA", "project": "C_TEAM", "name": "Anna"})
        self.assertIn("16:9", block)
        self.assertIn("Acme", block)
        self.assertIn("U_ANNA", block)

    def test_a_request_that_merely_starts_with_remember_is_a_request(self):
        out = self.mention("U_ANNA", "remember the red car from last week and redo it")
        self.assertEqual(out["action"], "turn")

    def test_queue_command_lists_the_line(self):
        self.mention("U_ANNA", "first", ts="9.1")
        self.mention("U_BEN", "second", ts="9.2")
        out = self.mention("U_ANNA", "queue", ts="9.3")
        self.assertEqual(out["action"], "command")
        self.assertIn("1. ", out["reply"])
        self.assertIn("U_BEN", out["reply"])


class DmStaysTheSameTest(ChannelCase):

    def test_a_dm_without_the_queue_option_still_uses_the_old_busy_rule(self):
        with mock.patch.object(self.b, "_live_turn", return_value=("rX", "other-conv")):
            out = self.b.route("U_ANNA", "another one")
        self.assertEqual(out["action"], "busy")
        self.assertEqual(self.started, [])

    def test_a_dm_start_gets_no_channel_extras(self):
        out = self.b.route("U_ANNA", "hello")
        self.assertEqual(out["action"], "turn")
        self.assertEqual(self.started[0]["extra"], {})

    def test_dm_does_not_need_a_mention(self):
        self.assertEqual(self.b.route("U_ANNA", "hello")["action"], "turn")

    def test_dm_approvals_for_an_unowned_conversation_are_left_to_the_panel(self):
        self.store.set("dm-conv", "D_ME", "1.0")
        with mock.patch("src.utils.tool_permissions.has_listener", return_value=True):
            self.assertIsNone(self.b.ask_approval("dm-conv", "run_script", "ls", 1))


class EventPlumbingTest(ChannelCase):

    def _ev(self, **kw):
        base = {"type": "app_mention", "user": "U_ANNA", "text": "<@U_BOT> make a cat",
                "channel": "C_TEAM", "ts": "800.1"}
        base.update(kw)
        return base

    def test_app_mention_is_actionable(self):
        self.assertTrue(is_actionable(self._ev()))
        self.assertFalse(is_actionable(self._ev(bot_id="B1")))

    def test_an_app_mention_runs_end_to_end(self):
        web = FakeClient()
        with tempfile.TemporaryDirectory() as d:
            _route_message(self.b, web, self._ev(), d)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.started[0]["text"], "make a cat")
        self.assertEqual(self.started[0]["extra"]["requester"]["name"], "Anna U_ANNA")
        self.assertIn(("+", "eyes"), self.client.reactions)

    def test_the_same_mention_arriving_twice_runs_once(self):
        web = FakeClient()
        with tempfile.TemporaryDirectory() as d:
            _route_message(self.b, web, self._ev(), d)
            _route_message(self.b, web, self._ev(type="message", channel_type="channel"), d)
        self.assertEqual(len(self.started), 1)

    def test_a_plain_channel_message_is_dropped_without_a_reaction(self):
        web = FakeClient()
        ev = {"type": "message", "channel_type": "channel", "user": "U_ANNA",
              "text": "anyone for lunch", "channel": "C_TEAM", "ts": "800.5"}
        with tempfile.TemporaryDirectory() as d:
            _route_message(self.b, web, ev, d)
        self.assertEqual(self.started, [])
        self.assertEqual(self.client.reactions, [])
        self.assertEqual(self.client.posted, [])

    def test_a_refused_stranger_is_told_in_the_thread(self):
        web = FakeClient()
        with tempfile.TemporaryDirectory() as d:
            _route_message(self.b, web, self._ev(user="U_STRANGER"), d)
        post = self.client.posted[-1]
        self.assertEqual(post["thread_ts"], "800.1")
        self.assertIn("Not authorised", post["text"])

    def test_the_downloads_of_a_request_land_in_its_own_folder(self):
        web = FakeClient()
        seen = {}

        def fake_dl(client, event, dest):
            seen["dest"] = Path(dest)
            return [], []

        with tempfile.TemporaryDirectory() as d, \
                mock.patch("src.utils.slack_bridge.download_files", fake_dl):
            _route_message(self.b, web, self._ev(files=[{"id": "F1"}]), d)
        self.assertEqual(seen["dest"].name, "C_TEAM_800.1")


class ParseDecisionTest(unittest.TestCase):

    def test_yes_and_no(self):
        for w in ("yes", "Yes!", " ok ", "approve", "go ahead", "y", "👍"):
            self.assertIs(parse_decision(w), True, w)
        for w in ("no", "No.", "deny", "stop", "n", "👎"):
            self.assertIs(parse_decision(w), False, w)

    def test_anything_else_is_neither(self):
        for w in ("maybe", "what does it do", "yes but only the first", ""):
            self.assertIsNone(parse_decision(w), w)


class ApprovalObjectTest(unittest.TestCase):
    def test_ids_are_distinct(self):
        self.assertNotEqual(_Approval("a").id, _Approval("a").id)


if __name__ == "__main__":
    unittest.main()
