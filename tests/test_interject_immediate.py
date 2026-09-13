"""A message sent mid-turn is read now, not when the turn ends.

Three places a message used to wait out, each pinned here:

* a **specialist delegation** — one long tool call to the orchestrator. The
  specialist is shown the message at its own next step (and the orchestrator
  still reads it afterwards, told the step already saw it);
* **after the last tool call** — no boundary left, so the turn goes back to read
  it instead of handing it to the next turn;
* the **render** — the orchestrator's loop has ended and ComfyUI runs for minutes.
  The executor's progress stream is watched, and a message wakes the orchestrator
  while the render carries on.

    python -m unittest discover -s tests
"""

import asyncio
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import src.agent as agents
from src.pipeline import Pipeline
from src.utils import interject_bus
from src.utils import interject_hook
from src.utils.interject_hook import InterjectHookProvider, SpecialistInterjectHook, take_pending
from src.utils.turn_checkin import interleave_checkins

PARTIALS = pathlib.Path(__file__).resolve().parent.parent / "config" / "system_prompts" / "orchestrator"


def _tool_event(text='{"status": "ok"}'):
    return SimpleNamespace(tool_use={"name": "get_node_schema", "toolUseId": "tu"},
                           result={"toolUseId": "tu", "status": "success",
                                   "content": [{"text": text}]})


def _said(event):
    return "\n".join(b.get("text", "") for b in event.result["content"])


class _OpenRun(unittest.TestCase):
    def setUp(self):
        interject_bus.open_run("r1", "t1")
        self.addCleanup(interject_bus.close_run, "r1")
        patcher = mock.patch.object(interject_hook, "_persist")
        self.persist = patcher.start()
        self.addCleanup(patcher.stop)


class ShowingWithoutTaking(_OpenRun):

    def test_a_specialist_is_shown_a_message_once(self):
        interject_bus.post("r1", "use 832x832")
        self.assertEqual([i["text"] for i in interject_bus.peek_for("generate_new_workflow")],
                         ["use 832x832"])
        self.assertEqual(interject_bus.peek_for("generate_new_workflow"), [])

    def test_showing_leaves_it_for_the_orchestrator(self):
        interject_bus.post("r1", "use 832x832")
        interject_bus.peek_for("generate_new_workflow")
        self.assertEqual(interject_bus.pending_count(), 1)

    def test_another_specialist_still_sees_it(self):
        interject_bus.post("r1", "use 832x832")
        interject_bus.peek_for("query_templates")
        self.assertEqual(len(interject_bus.peek_for("generate_new_workflow")), 1)

    def test_draining_says_who_was_shown(self):
        interject_bus.post("r1", "use 832x832")
        interject_bus.peek_for("generate_new_workflow")
        [item] = interject_bus.drain_detailed()
        self.assertEqual(item["relayed_to"], ["generate_new_workflow"])
        self.assertEqual(interject_bus.pending_count(), 0)

    def test_plain_drain_keeps_its_old_shape(self):
        interject_bus.post("r1", "hello")
        self.assertEqual(interject_bus.drain(), [{"text": "hello", "urgent": False}])


class TheSpecialistHook(_OpenRun):

    def test_the_specialist_reads_it_with_its_next_tool_result(self):
        interject_bus.post("r1", "make it portrait")
        event = _tool_event()
        SpecialistInterjectHook("generate_new_workflow")._on_after(event)
        self.assertIn("make it portrait", _said(event))
        self.assertIn('{"status": "ok"}', _said(event))

    def test_it_arrives_in_the_specialist_envelope(self):
        interject_bus.post("r1", "make it portrait")
        event = _tool_event()
        SpecialistInterjectHook("generate_new_workflow")._on_after(event)
        envelope = (PARTIALS / "interjection_specialist.md").read_text(encoding="utf-8").strip()
        self.assertIn(envelope.splitlines()[0], _said(event))

    def test_nothing_pending_leaves_the_result_alone(self):
        event = _tool_event()
        SpecialistInterjectHook("info")._on_after(event)
        self.assertEqual(event.result["content"], [{"text": '{"status": "ok"}'}])

    def test_it_is_not_recorded_until_the_orchestrator_reads_it(self):
        interject_bus.post("r1", "make it portrait")
        SpecialistInterjectHook("info")._on_after(_tool_event())
        self.persist.assert_not_called()

    def test_the_orchestrator_is_told_the_step_already_saw_it(self):
        interject_bus.post("r1", "make it portrait")
        SpecialistInterjectHook("generate_new_workflow")._on_after(_tool_event())
        event = _tool_event()
        InterjectHookProvider()._on_after(event)
        said = _said(event)
        self.assertIn("make it portrait", said)
        self.assertIn("`generate_new_workflow`", said)
        self.persist.assert_called_once()

    def test_specialists_carry_it_and_the_readers_do_not(self):
        for role in ("query_templates", "generate_new_workflow", "fix_workflow_assembly",
                     "info", "planner", "search_web"):
            self.assertIn(role, agents._SPECIALIST_INTERJECT_ROLES)
        for role in ("orchestrator", "vision_agent", "video_agent", "qa_checker"):
            self.assertNotIn(role, agents._SPECIALIST_INTERJECT_ROLES)


class ReadingItNow(_OpenRun):

    def test_nothing_said_is_nothing_to_read(self):
        self.assertIsNone(take_pending("interjection_after_answer"))

    def test_the_message_comes_back_in_the_envelope_asked_for(self):
        interject_bus.post("r1", "also a square one")
        text = take_pending("interjection_after_answer")
        envelope = (PARTIALS / "interjection_after_answer.md").read_text(encoding="utf-8")
        self.assertIn(envelope.strip().splitlines()[0], text)
        self.assertTrue(text.rstrip().endswith("also a square one"))

    def test_it_is_taken_and_recorded(self):
        interject_bus.post("r1", "also a square one")
        take_pending("interjection_during_render")
        self.assertEqual(interject_bus.pending_count(), 0)
        self.persist.assert_called_once()

    def test_every_envelope_exists(self):
        for name in ("interjection_specialist", "interjection_relayed",
                     "interjection_after_answer", "interjection_during_render"):
            self.assertTrue((PARTIALS / f"{name}.md").read_text(encoding="utf-8").strip(), name)


# ── the render: the stream is watched, and a message wakes the agent ─────────

def _collect(agen):
    async def run():
        return [item async for item in agen]
    return asyncio.run(run())


async def _slow_lines(*lines, delay=0.05):
    for line in lines:
        await asyncio.sleep(delay)
        yield line


class DuringTheRender(unittest.TestCase):

    def test_lines_pass_through_when_nothing_is_said(self):
        out = _collect(interleave_checkins(_slow_lines("a", "b"), lambda: False,
                                           lambda: _slow_lines(), poll=0.01))
        self.assertEqual(out, [("line", "a"), ("line", "b")])

    def test_a_message_wakes_the_agent_before_the_render_finishes(self):
        said = {"pending": True}

        async def checkin():
            said["pending"] = False
            yield {"data": "stopping it"}

        out = _collect(interleave_checkins(_slow_lines("step 1", "step 2", delay=0.1),
                                           lambda: said["pending"], checkin, poll=0.01))
        kinds = [kind for kind, _ in out]
        self.assertEqual(kinds.index("checkin"), 0)
        self.assertEqual([item for kind, item in out if kind == "line"], ["step 1", "step 2"])

    def test_the_render_keeps_going_while_the_agent_reads(self):
        """The check-in must not pause the stream it interrupts."""
        order = []
        said = {"pending": True}

        async def render():
            for i in range(3):
                await asyncio.sleep(0.03)
                order.append(f"render {i}")
                yield i

        async def checkin():
            said["pending"] = False
            await asyncio.sleep(0.08)
            order.append("agent answered")
            yield {"data": "ok"}

        _collect(interleave_checkins(render(), lambda: said["pending"], checkin, poll=0.01))
        self.assertLess(order.index("render 0"), order.index("agent answered"))

    def test_a_failed_check_in_does_not_end_the_render(self):
        said = {"pending": True}

        async def checkin():
            said["pending"] = False
            raise RuntimeError("model unreachable")
            yield  # pragma: no cover

        out = _collect(interleave_checkins(_slow_lines("a", "b"), lambda: said["pending"],
                                           checkin, poll=0.01))
        self.assertIn("model unreachable", out[0][1]["data"])
        self.assertEqual([item for kind, item in out if kind == "line"], ["a", "b"])

    def test_stopping_early_leaves_nothing_running(self):
        async def run():
            gen = interleave_checkins(_slow_lines("a", "b", "c"), lambda: False,
                                      lambda: _slow_lines(), poll=0.01)
            async for _kind, _item in gen:
                break
            await gen.aclose()
            await asyncio.sleep(0.02)
            return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        self.assertEqual(asyncio.run(run()), [])


class TheRenderCheckIn(_OpenRun):

    def test_the_agent_reads_the_message_and_its_answer_streams(self):
        seen = []

        async def stream_async(text):
            seen.append(text)
            yield {"data": "Interrupting the render."}

        fake = SimpleNamespace(_orchestrator_agent=SimpleNamespace(stream_async=stream_async))
        interject_bus.post("r1", "stop, wrong model")

        async def run():
            return [ev async for ev in Pipeline._render_checkin(fake)]
        events = asyncio.run(run())
        self.assertIn("stop, wrong model", seen[0])
        envelope = (PARTIALS / "interjection_during_render.md").read_text(encoding="utf-8")
        self.assertIn(envelope.strip().splitlines()[0], seen[0])
        self.assertIn("Interrupting the render.", "".join(e.get("data", "") for e in events))

    def test_with_nothing_said_the_agent_is_not_woken(self):
        fake = SimpleNamespace(_orchestrator_agent=SimpleNamespace(
            stream_async=mock.MagicMock(side_effect=AssertionError("should not run"))))

        async def run():
            return [ev async for ev in Pipeline._render_checkin(fake)]
        self.assertEqual(asyncio.run(run()), [])


if __name__ == "__main__":
    unittest.main()
