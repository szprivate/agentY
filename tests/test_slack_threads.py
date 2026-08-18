"""One conversation, one Slack thread — the panel's chat list, in a DM.

Everything used to land in whichever conversation the panel was in, flat in the
DM. Now a conversation IS a Slack thread: a root message naming it, every turn
as a reply underneath, and a reply of your own goes back to that conversation
rather than the current one. A message at the top level starts a fresh one, the
way a new chat does in the panel.

Slack has exactly one level of threading and the conversation takes it, so a
turn's working-out is one message that gets rewritten rather than a message per
tool. That is the trade this file pins down.

    python -m unittest discover -s tests
"""

import unittest
from unittest import mock

from src.utils import turn_bus
from src.utils.slack_bridge import SlackBridge, SlackTurn


class FakeClient:
    def __init__(self):
        self.posted, self.updated = [], []
        self._ts = 0

    def chat_postMessage(self, **kw):
        self._ts += 1
        self.posted.append(kw)
        return {"ts": f"ts{self._ts}"}

    def chat_update(self, **kw):
        self.updated.append(kw)
        return {"ok": True}


class _Store:
    """The binding table, in memory — the real one has its own tests."""

    def __init__(self, titles=None):
        self.bound = {}
        self.titles = titles or {}

    def set(self, tid, ch, ts):
        self.bound[(ch, ts)] = tid

    def get(self, tid):
        return next(({"channel": c, "root_ts": s}
                     for (c, s), v in self.bound.items() if v == tid), None)

    def reverse(self, ch, ts):
        return self.bound.get((ch, ts))

    def list(self, limit=200):
        return [{"id": k, "title": v} for k, v in self.titles.items()]


def _patch_store(case, store):
    for name, fn in (("set_slack_thread", store.set),
                     ("get_slack_thread", store.get),
                     ("thread_for_slack", store.reverse),
                     ("list_threads", store.list)):
        case.enterContext(mock.patch("src.utils.slack_bridge.cs." + name, fn))


class ThreadBindingTest(unittest.TestCase):

    def setUp(self):
        self.store = _Store({"conv1": "Samurai references"})
        self.client = FakeClient()
        self.b = SlackBridge(client=self.client, allowed_users=["U_ME"],
                             default_channel="D_ME")
        self.b.bot_user_id = "U_BOT"
        _patch_store(self, self.store)

    def test_a_conversation_opens_one_thread_named_after_it(self):
        ts = self.b.conversation_root("conv1", "make 6 refs")
        self.assertEqual(ts, "ts1")
        self.assertIn("Samurai references", self.client.posted[0]["text"])

    def test_the_same_conversation_reuses_its_thread(self):
        first = self.b.conversation_root("conv1")
        second = self.b.conversation_root("conv1")
        self.assertEqual(first, second)
        self.assertEqual(len(self.client.posted), 1, "it opened a second thread")

    def test_the_binding_outlives_this_process(self):
        """A restart that forgot it would open a second thread for a conversation
        that already has one, with nothing able to tell them apart afterwards."""
        self.b.conversation_root("conv1")
        self.assertEqual(self.store.reverse("D_ME", "ts1"), "conv1")

    def test_an_untitled_conversation_is_named_by_what_was_asked(self):
        self.b.conversation_root("conv_new", "cut this to five seconds")
        self.assertIn("cut this to five seconds", self.client.posted[0]["text"])

    def test_the_thread_follows_the_auto_title_when_it_lands(self):
        """Titles arrive a moment after the first turn; the heading has to catch up."""
        self.b.conversation_root("conv2", "make 6 refs")
        self.store.titles["conv2"] = "Kaiju night shots"
        self.b.conversation_root("conv2")
        self.b.flush()
        self.assertIn("Kaiju night shots", self.client.updated[-1]["text"])

    def test_it_does_not_rewrite_the_heading_for_nothing(self):
        """Asked on every turn; an edit each time is a round trip for no news."""
        self.b.conversation_root("conv1")
        self.b.conversation_root("conv1")
        self.b.flush()
        self.assertEqual(self.client.updated, [])

    def test_with_no_channel_there_is_no_thread_to_open(self):
        self.b.default_channel = ""
        self.assertEqual(self.b.conversation_root("conv1"), "")


class WhichConversationTest(unittest.TestCase):

    def setUp(self):
        self.store = _Store()
        self.started = []
        self.b = SlackBridge(client=FakeClient(), allowed_users=["U_ME"],
                             default_channel="D_ME",
                             start_turn=lambda text, files, tid="": (
                                 self.started.append((text, tid)) or "rid"),
                             interject=lambda rid, text: True)
        self.b.bot_user_id = "U_BOT"
        _patch_store(self, self.store)
        self.addCleanup(turn_bus._active.clear)

    def test_the_top_level_starts_a_new_conversation(self):
        self.b.route("U_ME", "render it")
        self.assertEqual(self.started, [("render it", "")])

    def test_a_reply_in_a_thread_continues_that_conversation(self):
        self.store.bound[("D_ME", "root9")] = "conv9"
        self.b.route("U_ME", "warmer", thread_ts="root9")
        self.assertEqual(self.started, [("warmer", "conv9")])

    def test_a_thread_nobody_knows_starts_a_conversation(self):
        """Better a fresh chat than somebody else's."""
        self.b.route("U_ME", "hello", thread_ts="root_unknown")
        self.assertEqual(self.started, [("hello", "")])


class OneTurnAtATimeTest(unittest.TestCase):
    """The pipeline is a singleton, and conversations make that visible.

    Before threads there was only one place a message could go, so a message
    during a run was always an interjection. Now it might be meant for a
    different chat entirely — and putting it into the running one would be worse
    than not delivering it at all.
    """

    def setUp(self):
        self.store = _Store()
        self.started, self.interjected = [], []
        self.b = SlackBridge(client=FakeClient(), allowed_users=["U_ME"],
                             default_channel="D_ME",
                             start_turn=lambda text, files, tid="": (
                                 self.started.append(tid) or "rid_new"),
                             interject=lambda rid, text: (
                                 self.interjected.append(rid) or True))
        self.b.bot_user_id = "U_BOT"
        _patch_store(self, self.store)
        self.addCleanup(turn_bus._active.clear)
        turn = turn_bus.Turn(request_id="rid_live", thread_id="conv_a")
        turn_bus._active["rid_live"] = turn
        self.b.turns["rid_live"] = SlackTurn(self.b, turn, "D_ME", "root_a")
        self.store.bound[("D_ME", "root_a")] = "conv_a"
        self.store.bound[("D_ME", "root_b")] = "conv_b"

    def test_a_reply_in_the_running_conversation_steers_it(self):
        out = self.b.route("U_ME", "actually, warmer", thread_ts="root_a")
        self.assertEqual(out["action"], "interject")
        self.assertEqual(self.interjected, ["rid_live"])

    def test_a_reply_in_ANOTHER_conversation_is_not_put_into_the_running_one(self):
        out = self.b.route("U_ME", "unrelated question", thread_ts="root_b")
        self.assertEqual(out["action"], "busy")
        self.assertEqual(self.interjected, [], "it went into the wrong conversation")
        self.assertEqual(self.started, [], "and it must not run two turns at once")

    def test_a_new_top_level_message_during_a_run_is_refused_not_run_alongside(self):
        out = self.b.route("U_ME", "something new", thread_ts="")
        self.assertEqual(out["action"], "busy")
        self.assertEqual(self.started, [])

    def test_the_refusal_names_what_it_is_busy_with(self):
        out = self.b.route("U_ME", "unrelated", thread_ts="root_b")
        self.assertEqual(out["running_thread"], "conv_a")


class AskScopeTest(unittest.TestCase):
    """A question belongs to the conversation that asked it."""

    def setUp(self):
        self.answered = []
        self.b = SlackBridge(client=FakeClient(), allowed_users=["U_ME"],
                             default_channel="D_ME",
                             start_turn=lambda text, files, tid="": "rid_new",
                             answer=lambda rid, text: (
                                 self.answered.append(rid) or True))
        self.b.bot_user_id = "U_BOT"
        self.store = _Store()
        _patch_store(self, self.store)
        turn = turn_bus.Turn(request_id="rid_ask", thread_id="conv_a")
        st = SlackTurn(self.b, turn, "D_ME", "root_a")
        st.ask_request_id = "rid_ask"
        self.b.turns["rid_ask"] = st
        self.store.bound[("D_ME", "root_a")] = "conv_a"
        self.store.bound[("D_ME", "root_b")] = "conv_b"
        self.addCleanup(turn_bus._active.clear)

    def test_replying_in_that_thread_answers_it(self):
        out = self.b.route("U_ME", "yes, retry", thread_ts="root_a")
        self.assertEqual(out["action"], "answer")
        self.assertEqual(self.answered, ["rid_ask"])

    def test_a_message_in_another_conversation_is_not_taken_as_the_answer(self):
        """Answering with words written for a different chat is worse than
        leaving the question open."""
        out = self.b.route("U_ME", "how much did that cost?", thread_ts="root_b")
        self.assertNotEqual(out["action"], "answer")
        self.assertEqual(self.answered, [])


if __name__ == "__main__":
    unittest.main()
