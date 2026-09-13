"""A conversation the agent still remembers after the host restarts.

The orchestrator's message list was kept in exactly one place — a dict in the
running process — so a restart wiped it, and the next message in an old
conversation reached an agent that had never heard of it. The user had to remind
it, every time. These hold the three pieces of the fix: the messages are saved in
a form that survives, they come back when the process has none, and a
conversation saved before any of this comes back from its transcript.

    python -m unittest discover -s tests
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import agentY_server as srv
from src.utils.brain_memory import (
    MAX_TOOL_TEXT,
    RESTORED_NOTE,
    choose_history,
    serialize_messages,
    transcript_to_messages,
)


def text(t):
    return {"text": t}


def rows(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]


LIVE = [
    {"role": "user", "content": [
        text("make a cat"),
        {"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n..."}}}]},
    {"role": "assistant", "content": [
        {"reasoningContent": {"reasoningText": {"text": "thinking", "signature": "sig"}}},
        text("On it."),
        {"toolUse": {"toolUseId": "call_1", "name": "prepare_workflow",
                     "input": {"request": "a cat"}}}]},
    {"role": "user", "content": [
        {"toolResult": {"toolUseId": "call_1", "status": "success",
                        "content": [text("x" * (MAX_TOOL_TEXT + 500))]}}]},
    {"role": "assistant", "content": [text("Here is your cat.")]},
]


class SavingTheMessages(unittest.TestCase):

    def setUp(self):
        self.saved = serialize_messages(LIVE)

    def test_the_result_survives_json(self):
        self.assertEqual(json.loads(json.dumps(self.saved)), self.saved)

    def test_every_message_keeps_its_place(self):
        self.assertEqual([m["role"] for m in self.saved],
                         ["user", "assistant", "user", "assistant"])

    def test_an_image_becomes_a_line_saying_it_was_there(self):
        blocks = self.saved[0]["content"]
        self.assertEqual(blocks[0], text("make a cat"))
        self.assertIn("image was attached", blocks[1]["text"])

    def test_a_tool_call_and_its_result_stay_paired(self):
        use = self.saved[1]["content"][-1]["toolUse"]
        result = self.saved[2]["content"][0]["toolResult"]
        self.assertEqual(use["toolUseId"], result["toolUseId"])
        self.assertEqual(use["input"], {"request": "a cat"})
        self.assertEqual(result["status"], "success")

    def test_reasoning_is_not_kept(self):
        """A reasoning signature is valid only for the model that wrote it."""
        self.assertNotIn("reasoningContent", json.dumps(self.saved))

    def test_long_tool_output_is_clipped(self):
        out = self.saved[2]["content"][0]["toolResult"]["content"][0]["text"]
        self.assertLess(len(out), MAX_TOOL_TEXT + 100)
        self.assertIn("not kept", out)

    def test_bytes_hidden_in_a_tool_input_do_not_break_it(self):
        saved = serialize_messages([{"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "c", "name": "t", "input": {"blob": b"\x00\x01"}}}]}])
        json.dumps(saved)

    def test_a_message_emptied_of_everything_stays_as_a_placeholder(self):
        saved = serialize_messages([{"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "hm"}}}]}])
        self.assertEqual(saved, [{"role": "assistant", "content": [text("(empty)")]}])

    def test_things_that_are_not_messages_are_ignored(self):
        self.assertEqual(serialize_messages([None, "x", {"role": "system", "content": "y"}]), [])
        self.assertEqual(serialize_messages(None), [])


class RebuildingFromTheTranscript(unittest.TestCase):

    def test_the_message_being_answered_is_not_repeated_as_history(self):
        history = transcript_to_messages(rows(("user", "a cat"), ("assistant", "here"),
                                              ("user", "make it warmer")))
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertNotIn("warmer", json.dumps(history))

    def test_it_says_it_was_restored(self):
        history = transcript_to_messages(rows(("user", "a cat"), ("assistant", "here")))
        first = history[0]["content"][0]["text"]
        self.assertTrue(first.startswith(RESTORED_NOTE))
        self.assertTrue(first.endswith("a cat"))

    def test_slash_commands_are_not_conversation(self):
        history = transcript_to_messages(rows(("user", "/images"), ("user", "a cat"),
                                              ("assistant", "here")))
        self.assertNotIn("/images", json.dumps(history))

    def test_one_side_speaking_twice_is_one_message(self):
        history = transcript_to_messages(rows(("user", "a cat"), ("user", "orange"),
                                              ("assistant", "here")))
        self.assertEqual(len(history), 2)
        self.assertIn("a cat\n\norange", history[0]["content"][0]["text"])

    def test_the_newest_exchanges_win_the_budget(self):
        pairs = []
        for i in range(30):
            pairs += [("user", f"ask {i}"), ("assistant", f"answer {i}")]
        history = transcript_to_messages(rows(*pairs), max_messages=6)
        self.assertEqual(len(history), 6)
        self.assertTrue(history[0]["content"][0]["text"].endswith("ask 27"))
        self.assertEqual(history[-1]["content"][0]["text"], "answer 29")

    def test_it_never_begins_with_the_agent(self):
        history = transcript_to_messages(rows(("assistant", "hello"), ("user", "a cat"),
                                              ("assistant", "here")))
        self.assertEqual(history[0]["role"], "user")

    def test_nothing_answered_yet_is_no_history(self):
        self.assertEqual(transcript_to_messages(rows(("user", "a cat"))), [])
        self.assertEqual(transcript_to_messages(None), [])


class WhichHistoryWins(unittest.TestCase):

    SAVED = [{"role": "user", "content": [text("saved")]}]
    ROWS = rows(("user", "transcript"), ("assistant", "reply"))

    def test_this_processs_own_copy_first(self):
        live = [{"role": "user", "content": [text("live")]}]
        self.assertEqual(choose_history(live, self.SAVED, self.ROWS), (live, "cache"))

    def test_then_the_saved_state(self):
        self.assertEqual(choose_history(None, self.SAVED, self.ROWS), (self.SAVED, "saved"))

    def test_then_the_transcript(self):
        history, source = choose_history(None, None, self.ROWS)
        self.assertEqual(source, "transcript")
        self.assertEqual(len(history), 2)

    def test_a_new_conversation_has_none(self):
        self.assertEqual(choose_history(None, None, rows(("user", "hi"))), ([], "none"))


# ── through the real restore / save path ─────────────────────────────────────

class _Pipeline:
    def __init__(self, messages=None):
        self._orchestrator_agent = SimpleNamespace(messages=list(messages or []))
        self._session = None
        self._last_brainbriefing_json = None
        self._last_prior_summary = None


class ResumingAfterARestart(unittest.TestCase):
    """The reported bug, end to end: an old conversation, a process with no memory."""

    def setUp(self):
        self.cache = {}
        for target, name, value in ((srv, "_thread_brain_cache", self.cache),
                                    (srv.cs, "get_gallery", lambda tid: [])):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def restore(self, state, transcript=()):
        pipe = _Pipeline()
        with mock.patch.object(srv.cs, "load_state", return_value=state), \
                mock.patch.object(srv.cs, "get_thread",
                                  return_value={"messages": list(transcript)}):
            srv._restore_state(pipe, "t1")
        return pipe._orchestrator_agent.messages

    def test_the_saved_conversation_comes_back_when_the_process_has_none(self):
        saved = serialize_messages(LIVE)
        self.assertEqual(self.restore({"brain_messages": saved}), saved)

    def test_a_conversation_saved_before_this_existed_comes_back_from_its_transcript(self):
        messages = self.restore({"brain_messages": None},
                                rows(("user", "a cat"), ("assistant", "here"),
                                     ("user", "warmer")))
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        self.assertTrue(messages[0]["content"][0]["text"].startswith(RESTORED_NOTE))

    def test_with_no_saved_state_at_all_the_transcript_still_works(self):
        messages = self.restore(None, rows(("user", "a cat"), ("assistant", "here"),
                                           ("user", "warmer")))
        self.assertEqual(len(messages), 2)

    def test_within_one_run_the_live_copy_wins(self):
        live = [{"role": "user", "content": [text("live")]}]
        self.cache["t1"] = live
        self.assertEqual(self.restore({"brain_messages": serialize_messages(LIVE)}), live)

    def test_saving_writes_the_messages_to_disk(self):
        with mock.patch.object(srv.cs, "save_state") as save:
            srv._save_state(_Pipeline(LIVE), "t1")
        saved = save.call_args.kwargs["brain_messages"]
        self.assertEqual(saved, serialize_messages(LIVE))
        json.dumps(saved)


if __name__ == "__main__":
    unittest.main()
