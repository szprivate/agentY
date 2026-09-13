"""Undo one agent step: the conversation, the agent's memory, and the canvas.

Each turn writes a checkpoint as it starts — where the transcript and the image
list stood, what the agent remembered, and the canvas as the panel sent it. Undo
pops the newest and puts those back; the canvas half is the panel's, and the
event it gets carries what it needs to tell whether restoring is safe.

    python -m unittest discover -s tests
"""

import os
import queue
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from src.utils import agentY_server as srv
from src.utils import conversation_store as cs


class TempStore(unittest.TestCase):
    """A throwaway conversation database for each test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "AGENTY_CONVERSATION_DB": os.path.join(tmp.name, "conversations.sqlite")})
        env.start()
        self.addCleanup(env.stop)
        cs._INITIALISED = False
        self.addCleanup(setattr, cs, "_INITIALISED", False)

    def conversation(self, *pairs):
        tid = cs.create_thread()
        for role, text in pairs:
            cs.add_message(tid, role, text)
        return tid

    def texts(self, tid):
        return [m["content"] for m in cs.get_thread(tid)["messages"]]


class WhereATurnBegins(TempStore):

    def test_undo_takes_the_turn_and_the_message_that_started_it(self):
        tid = self.conversation(("user", "a cat"), ("assistant", "here is a cat"),
                                ("user", "make it warmer"))
        cs.push_checkpoint(tid, turn_text="make it warmer")
        cs.add_message(tid, "assistant", "warmer now")
        cs.add_gallery_image(tid, "/out/warm.png")
        cp = cs.pop_checkpoint(tid)
        removed = cs.rewind_thread(tid, cp["transcript_cut"], cp["gallery_cut"])
        self.assertEqual(self.texts(tid), ["a cat", "here is a cat"])
        self.assertEqual(removed, (2, 1))

    def test_earlier_outputs_stay_in_the_image_list(self):
        tid = self.conversation(("user", "one"))
        cs.add_gallery_image(tid, "/out/1.png")
        cs.push_checkpoint(tid, turn_text="one")
        cs.add_gallery_image(tid, "/out/2.png")
        cp = cs.pop_checkpoint(tid)
        cs.rewind_thread(tid, cp["transcript_cut"], cp["gallery_cut"])
        self.assertEqual([g["path"] for g in cs.get_gallery(tid)], ["/out/1.png"])

    def test_a_replayed_message_keeps_what_was_already_there(self):
        """/resend runs an old message without saving a new one."""
        tid = self.conversation(("user", "a cat"), ("assistant", "cat"), ("user", "/resend"))
        cs.push_checkpoint(tid, turn_text="a cat")
        cs.add_message(tid, "assistant", "another cat")
        cp = cs.pop_checkpoint(tid)
        cs.rewind_thread(tid, cp["transcript_cut"], cp["gallery_cut"])
        self.assertEqual(self.texts(tid), ["a cat", "cat", "/resend"])

    def test_a_message_sent_into_the_running_turn_goes_with_it(self):
        tid = self.conversation(("user", "render it"))
        cs.push_checkpoint(tid, turn_text="render it")
        cs.add_message(tid, "user", "use 30 steps")
        cs.add_message(tid, "assistant", "done")
        cp = cs.pop_checkpoint(tid)
        cs.rewind_thread(tid, cp["transcript_cut"], cp["gallery_cut"])
        self.assertEqual(self.texts(tid), [])


class OneStepAtATime(TempStore):

    def test_the_newest_step_comes_off_first(self):
        tid = self.conversation(("user", "first"))
        first = cs.push_checkpoint(tid, label="first")
        second = cs.push_checkpoint(tid, label="second")
        self.assertEqual(cs.pop_checkpoint(tid)["id"], second)
        self.assertEqual(cs.pop_checkpoint(tid)["id"], first)
        self.assertIsNone(cs.pop_checkpoint(tid))

    def test_only_the_newest_few_are_kept(self):
        tid = self.conversation()
        for i in range(5):
            cs.push_checkpoint(tid, label=str(i), keep=3)
        self.assertEqual(cs.count_checkpoints(tid), 3)
        self.assertEqual(cs.pop_checkpoint(tid)["label"], "4")

    def test_what_was_saved_comes_back_as_it_was(self):
        tid = self.conversation()
        brain = [{"role": "user", "content": [{"text": "hi"}]}]
        cs.push_checkpoint(tid, brain_messages=brain, agent_session={"session_id": tid},
                           last_brainbriefing='{"a": 1}',
                           canvas_graph={"nodes": [], "links": []},
                           canvas_before_hash="abc:10", canvas_workflow="portrait.json")
        cp = cs.pop_checkpoint(tid)
        self.assertEqual(cp["brain_messages"], brain)
        self.assertEqual(cp["agent_session"], {"session_id": tid})
        self.assertEqual(cp["last_brainbriefing"], '{"a": 1}')
        self.assertEqual(cp["canvas_graph"], {"nodes": [], "links": []})
        self.assertEqual((cp["canvas_before_hash"], cp["canvas_workflow"]),
                         ("abc:10", "portrait.json"))

    def test_the_canvas_at_the_end_of_the_turn_is_recorded(self):
        tid = self.conversation()
        cp_id = cs.push_checkpoint(tid)
        self.assertTrue(cs.set_checkpoint_canvas_after(tid, cp_id, "def:12"))
        self.assertFalse(cs.set_checkpoint_canvas_after("another-conversation", cp_id, "x"))
        self.assertEqual(cs.pop_checkpoint(tid)["canvas_after_hash"], "def:12")

    def test_deleting_the_conversation_deletes_its_checkpoints(self):
        tid = self.conversation()
        cs.push_checkpoint(tid)
        cs.delete_thread(tid)
        cs.create_thread(thread_id=tid)
        self.assertEqual(cs.count_checkpoints(tid), 0)


class TheUndo(TempStore):

    BEFORE = [{"role": "user", "content": [{"text": "a cat"}]},
              {"role": "assistant", "content": [{"text": "a cat"}]}]

    def setUp(self):
        super().setUp()
        self.cache = {}
        self.registry = {}
        for name, value in (("_thread_brain_cache", self.cache),
                            ("_run_registry", self.registry)):
            patcher = mock.patch.object(srv, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def step(self):
        tid = self.conversation(("user", "a cat"), ("assistant", "a cat"),
                                ("user", "make it warmer"))
        cs.push_checkpoint(tid, turn_text="make it warmer", label="make it warmer",
                           brain_messages=self.BEFORE, agent_session={"session_id": tid},
                           canvas_graph={"nodes": [{"id": 1}], "links": []},
                           canvas_before_hash="aaa:1", canvas_workflow="cat.json")
        cs.add_message(tid, "assistant", "warmer")
        cs.save_state(tid, brain_messages=self.BEFORE + [
            {"role": "user", "content": [{"text": "make it warmer"}]},
            {"role": "assistant", "content": [{"text": "warmer"}]}])
        return tid

    def test_the_conversation_and_the_agent_forget_the_step(self):
        tid = self.step()
        [event] = srv._undo_last_step(tid)
        self.assertEqual(event["type"], "undo")
        self.assertEqual(self.texts(tid), ["a cat", "a cat"])
        self.assertEqual(cs.load_state(tid)["brain_messages"], self.BEFORE)
        self.assertEqual(self.cache[tid], self.BEFORE)

    def test_the_panel_is_given_the_canvas_to_put_back(self):
        tid = self.step()
        [event] = srv._undo_last_step(tid)
        self.assertEqual(event["canvas_graph"], {"nodes": [{"id": 1}], "links": []})
        self.assertEqual((event["canvas_before_hash"], event["canvas_workflow"]),
                         ("aaa:1", "cat.json"))
        self.assertIn("make it warmer", event["message"])

    def test_the_saved_panel_is_dropped_so_it_is_rebuilt(self):
        tid = self.step()
        cs.save_panel(tid, "<div>the undone step</div>")
        srv._undo_last_step(tid)
        self.assertEqual(cs.get_panel(tid), "")

    def test_nothing_to_undo_says_so(self):
        tid = self.conversation(("user", "hi"))
        [event] = srv._undo_last_step(tid)
        self.assertEqual(event["type"], "system")
        self.assertIn("Nothing to undo", event["data"])

    def test_it_will_not_undo_under_a_running_turn(self):
        tid = self.step()
        self.registry["r1"] = {"thread_id": tid}
        [event] = srv._undo_last_step(tid)
        self.assertEqual(event["type"], "system")
        self.assertEqual(cs.count_checkpoints(tid), 1)

    def test_a_turn_in_another_conversation_does_not_block_it(self):
        tid = self.step()
        self.registry["r1"] = {"thread_id": "somewhere-else"}
        [event] = srv._undo_last_step(tid)
        self.assertEqual(event["type"], "undo")

    def test_from_slack_the_open_panel_is_told(self):
        tid = self.step()
        with mock.patch.object(srv.notify_bus, "emit") as emit:
            srv._undo_last_step(tid, origin="slack")
        self.assertIn("undo", [c.args[0].get("kind") for c in emit.call_args_list])

    def test_the_slash_command_reaches_it(self):
        tid = self.step()
        [event] = srv._handle_command(tid, "/undo")
        self.assertEqual(event["type"], "undo")

    def test_slack_takes_the_bare_word(self):
        """Slack's composer eats a leading slash as one of its own commands."""
        for said in ("undo", "/undo", "  Undo "):
            self.assertTrue(srv._is_undo_request(said), said)
        for said in ("undo that last change", "don't undo", ""):
            self.assertFalse(srv._is_undo_request(said), said)


# ── the checkpoint a turn leaves ─────────────────────────────────────────────

class _Session:
    def __init__(self):
        self.current_output_paths = []
        self.last_user_input_images = []
        self.session_id = "t1"

    def model_dump(self):
        return {"session_id": self.session_id}


class _TurnPipeline:
    def __init__(self):
        self._session = _Session()
        self._last_brainbriefing_json = None
        self._last_prior_summary = None
        self._orchestrator_agent = SimpleNamespace(messages=[
            {"role": "user", "content": [{"text": "earlier"}]},
            {"role": "assistant", "content": [{"text": "sure"}]}])

    async def stream_async(self, content, **kw):
        yield {"data": "warmer now"}

    async def _await_pending_compression(self):
        return None


def run_turn(push):
    out_q: "queue.Queue" = queue.Queue()
    with ExitStack() as stack:
        for target, name, value in (
                (srv, "_agent_ref", _TurnPipeline()),
                (srv, "_restore_state", lambda *a, **k: None),
                (srv, "_save_state", lambda *a, **k: None),
                (srv, "_resolve_qa_briefing", lambda *a, **k: None),
                (srv.cs, "get_thread",
                 lambda tid: {"id": tid, "messages": [{"role": "assistant", "content": "hi"}]}),
                (srv.cs, "add_message", lambda *a, **k: 1),
                (srv.cs, "push_checkpoint", push)):
            stack.enter_context(mock.patch.object(target, name, value))
        srv._run_pipeline_turn("t1", "make it warmer", [], out_q, "rid1", {"emitted": False},
                               origin="panel", canvas_graph={"nodes": [], "links": []},
                               canvas_hash="abc:2", canvas_workflow="cat.json")
    events = []
    while True:
        ev = out_q.get_nowait()
        if ev is None:
            break
        events.append(ev)
    return events


class ATurnLeavesACheckpoint(unittest.TestCase):

    def test_the_panel_is_told_which_checkpoint_the_turn_is(self):
        events = run_turn(mock.MagicMock(return_value=7))
        kinds = [e.get("type") for e in events]
        checkpoint = next(e for e in events if e.get("type") == "checkpoint")
        self.assertEqual((checkpoint["id"], checkpoint["thread_id"], checkpoint["text"]),
                         (7, "t1", "make it warmer"))
        self.assertLess(kinds.index("checkpoint"), kinds.index("text"))

    def test_it_records_the_state_the_turn_starts_from(self):
        push = mock.MagicMock(return_value=7)
        run_turn(push)
        self.assertEqual(push.call_args.args, ("t1",))
        kw = push.call_args.kwargs
        self.assertEqual(kw["turn_text"], "make it warmer")
        self.assertEqual(kw["canvas_graph"], {"nodes": [], "links": []})
        self.assertEqual((kw["canvas_before_hash"], kw["canvas_workflow"]),
                         ("abc:2", "cat.json"))
        self.assertEqual([m["role"] for m in kw["brain_messages"]], ["user", "assistant"])
        self.assertEqual(kw["agent_session"], {"session_id": "t1"})

    def test_a_checkpoint_that_fails_does_not_stop_the_turn(self):
        events = run_turn(mock.MagicMock(side_effect=RuntimeError("disk full")))
        self.assertNotIn("checkpoint", [e.get("type") for e in events])
        said = "".join(e.get("data", "") for e in events if e.get("type") == "text")
        self.assertIn("warmer now", said)


if __name__ == "__main__":
    unittest.main()
