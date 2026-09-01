"""Tests for mid-run interjections — talking to a turn that is already running.

Two halves:

* the mailbox (:mod:`src.utils.interject_bus`): one active run at a time, posts
  from another thread, and everything undelivered handed back at close so a
  message that arrived a moment too late is queued rather than lost;
* the delivery hook (:mod:`src.utils.interject_hook`): normal messages ride out
  with the result of the tool that just ran, urgent ones cancel the pending call
  so the model reads them instead. Both paths must reach the model wrapped in the
  guidance from config/system_prompts/orchestrator/interjection*.md — an
  interjection that arrives looking like the tool's own output is worse than one
  that never arrives.

    python -m unittest discover -s tests
"""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from src.utils import interject_bus
from src.utils.interject_hook import InterjectHookProvider


def _tool_event(name="prepare_workflow", text="{\"status\": \"ok\"}"):
    """An AfterToolCallEvent-alike: only the fields the hook touches."""
    return SimpleNamespace(
        tool_use={"name": name, "toolUseId": "tu-1"},
        result={"toolUseId": "tu-1", "status": "success", "content": [{"text": text}]},
        exception=None,
    )


def _before_event(name="run_workflow_now"):
    return SimpleNamespace(tool_use={"name": name, "toolUseId": "tu-2"}, cancel_tool=False)


def _delivered_text(event) -> str:
    return "\n".join(c.get("text", "") for c in event.result["content"])


class MailboxTest(unittest.TestCase):
    def setUp(self):
        interject_bus.close_run(interject_bus.active_run() or "")
        self.addCleanup(lambda: interject_bus.close_run(interject_bus.active_run() or ""))

    def test_nothing_to_interject_into_when_no_turn_is_running(self):
        self.assertFalse(interject_bus.post("run-1", "hold on"))
        self.assertEqual(interject_bus.pending_count(), 0)

    def test_a_stale_request_id_is_refused(self):
        interject_bus.open_run("run-2")
        self.assertFalse(interject_bus.post("run-1", "from the previous turn"))
        self.assertTrue(interject_bus.post("run-2", "for this one"))

    def test_empty_text_is_refused(self):
        interject_bus.open_run("run-1")
        self.assertFalse(interject_bus.post("run-1", "   "))

    def test_messages_come_out_in_the_order_they_were_sent(self):
        interject_bus.open_run("run-1")
        interject_bus.post("run-1", "first")
        interject_bus.post("run-1", "second")
        self.assertEqual([i["text"] for i in interject_bus.drain()], ["first", "second"])
        self.assertEqual(interject_bus.pending_count(), 0, "drain must clear the mailbox")

    def test_close_hands_back_what_was_never_delivered(self):
        interject_bus.open_run("run-1")
        interject_bus.post("run-1", "too late")
        self.assertEqual(interject_bus.close_run("run-1"), ["too late"])
        self.assertIsNone(interject_bus.active_run())

    def test_opening_a_run_drops_anything_left_from_the_last_one(self):
        interject_bus.open_run("run-1")
        interject_bus.post("run-1", "stale")
        interject_bus.open_run("run-2")
        self.assertEqual(interject_bus.pending_count(), 0)

    def test_the_thread_rides_along_for_the_delivering_side(self):
        interject_bus.open_run("run-1", "thread-42")
        self.assertEqual(interject_bus.thread_id(), "thread-42")
        interject_bus.close_run("run-1")
        self.assertEqual(interject_bus.thread_id(), "")

    def test_posting_from_another_thread_is_safe(self):
        # The real caller is a Flask request thread while the turn runs its own
        # loop in a different one.
        interject_bus.open_run("run-1")
        errors = []

        def post(n):
            try:
                for i in range(50):
                    interject_bus.post("run-1", f"{n}-{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=post, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(interject_bus.drain()), 200)


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        interject_bus.close_run(interject_bus.active_run() or "")
        self.addCleanup(lambda: interject_bus.close_run(interject_bus.active_run() or ""))
        self.hook = InterjectHookProvider()
        # Delivery persists into the conversation; the store is not under test.
        patcher = mock.patch("src.utils.conversation_store.add_message")
        self.add_message = patcher.start()
        self.addCleanup(patcher.stop)

    def test_registers_on_both_tool_boundaries(self):
        registry = mock.Mock()
        self.hook.register_hooks(registry)
        events = [c.args[0].__name__ for c in registry.add_callback.call_args_list]
        self.assertIn("BeforeToolCallEvent", events)
        self.assertIn("AfterToolCallEvent", events)

    # ── normal: rides out with the result of the call that just finished ─────
    def test_the_message_reaches_the_model_with_the_tool_result(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "make it colder, less orange")
        ev = _tool_event()
        self.hook._on_after(ev)
        text = _delivered_text(ev)
        self.assertIn("make it colder, less orange", text)
        self.assertIn('{"status": "ok"}', text, "the tool's own result must survive")
        self.assertEqual(interject_bus.pending_count(), 0)

    def test_it_arrives_wrapped_in_the_guidance_partial(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "use the second reference")
        ev = _tool_event()
        self.hook._on_after(ev)
        text = _delivered_text(ev)
        self.assertIn("USER INTERJECTION", text)
        # The wording lives in the .md, not in the code — check the file's own
        # instruction reached the model rather than asserting on a literal here.
        from src.pipeline import _orch_partial
        self.assertIn(_orch_partial("interjection"), text)
        self.assertNotEqual(_orch_partial("interjection"), "", "the partial must exist")

    def test_several_messages_are_delivered_together_in_order(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "first thought")
        interject_bus.post("run-1", "second thought")
        ev = _tool_event()
        self.hook._on_after(ev)
        text = _delivered_text(ev)
        self.assertLess(text.index("first thought"), text.index("second thought"))

    def test_an_untouched_result_when_nothing_was_sent(self):
        interject_bus.open_run("run-1", "t1")
        ev = _tool_event()
        before = ev.result
        self.hook._on_after(ev)
        self.assertIs(ev.result, before, "no interjection, no rewrite")

    def test_delivery_is_recorded_in_the_conversation(self):
        interject_bus.open_run("run-1", "thread-9")
        interject_bus.post("run-1", "keep the wide shot")
        self.hook._on_after(_tool_event())
        self.add_message.assert_called_once_with("thread-9", "user", "keep the wide shot")

    def test_an_undelivered_message_is_never_recorded(self):
        interject_bus.open_run("run-1", "thread-9")
        interject_bus.post("run-1", "too late")
        interject_bus.close_run("run-1")
        self.add_message.assert_not_called()

    # ── urgent: cancel the pending call so it is read first ──────────────────
    def test_urgent_cancels_the_tool_and_carries_the_message(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "stop, wrong model", urgent=True)
        ev = _before_event()
        self.hook._on_before(ev)
        self.assertIsInstance(ev.cancel_tool, str)
        self.assertIn("stop, wrong model", ev.cancel_tool)
        from src.pipeline import _orch_partial
        self.assertIn(_orch_partial("interjection_urgent"), ev.cancel_tool)
        self.assertEqual(interject_bus.pending_count(), 0)

    def test_a_normal_message_does_not_cancel_anything(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "fyi, prefer 16:9")
        ev = _before_event()
        self.hook._on_before(ev)
        self.assertFalse(ev.cancel_tool)
        self.assertEqual(interject_bus.pending_count(), 1, "it waits for the after-hook")

    def test_an_urgent_message_takes_the_normal_ones_with_it(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "one more thing")
        interject_bus.post("run-1", "actually stop", urgent=True)
        ev = _before_event()
        self.hook._on_before(ev)
        self.assertIn("one more thing", ev.cancel_tool)
        self.assertIn("actually stop", ev.cancel_tool)

    # ── a hook must never take the turn down ─────────────────────────────────
    def test_a_broken_result_shape_is_survived(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "hello")
        ev = SimpleNamespace(tool_use={"name": "x"}, result="not a dict", exception=None)
        self.hook._on_after(ev)          # must not raise
        self.assertEqual(ev.result, "not a dict")

    def test_a_failure_inside_the_hook_is_swallowed(self):
        interject_bus.open_run("run-1", "t1")
        interject_bus.post("run-1", "hello")
        with mock.patch("src.utils.interject_hook._format", side_effect=RuntimeError("boom")):
            self.hook._on_after(_tool_event())   # must not raise

    def test_the_orchestrator_actually_carries_the_hook(self):
        """Everything above is moot if the provider isn't on the live agent."""
        import inspect

        from src import agent as agent_mod
        src = inspect.getsource(agent_mod.create_orchestrator_agent)
        self.assertIn("InterjectHookProvider()", src)


class RouteTest(unittest.TestCase):
    """POST /agentY/interject, against the real Flask app."""

    @classmethod
    def setUpClass(cls):
        from src.utils.agentY_server import _build_app
        from tests.route_client import authorised_client
        cls.client = authorised_client(_build_app())

    def setUp(self):
        interject_bus.close_run(interject_bus.active_run() or "")
        self.addCleanup(lambda: interject_bus.close_run(interject_bus.active_run() or ""))

    def _post(self, **body):
        return self.client.post("/agentY/interject", json=body)

    def test_409_when_no_turn_is_running(self):
        # The panel reads this as "leave it queued, it sends next" — the message
        # must not be silently accepted into nothing.
        self.assertEqual(self._post(request_id="r1", text="hi").status_code, 409)

    def test_accepted_while_a_turn_is_running(self):
        interject_bus.open_run("r1", "t1")
        res = self._post(request_id="r1", text="make it colder")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])
        self.assertEqual([i["text"] for i in interject_bus.drain()], ["make it colder"])

    def test_the_urgent_flag_is_carried_through(self):
        interject_bus.open_run("r1", "t1")
        self.assertTrue(self._post(request_id="r1", text="stop", urgent=True).get_json()["urgent"])
        self.assertTrue(interject_bus.has_urgent())

    def test_empty_text_is_rejected(self):
        interject_bus.open_run("r1", "t1")
        self.assertEqual(self._post(request_id="r1", text="  ").status_code, 400)

    def test_a_request_id_from_an_earlier_turn_is_rejected(self):
        interject_bus.open_run("r2", "t1")
        self.assertEqual(self._post(request_id="r1", text="late").status_code, 409)

    def test_the_route_does_not_persist_only_delivery_does(self):
        interject_bus.open_run("r1", "t1")
        with mock.patch("src.utils.conversation_store.add_message") as add:
            self._post(request_id="r1", text="hi")
        add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
