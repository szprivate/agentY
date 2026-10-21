"""A turn must not stop for the wrong reasons (issues #67 and #68).

#67: the model hit its output limit, or the prompt no longer fit, and the turn
ended with an error - the task waited for a nudge. It is now trimmed and re-driven,
a bounded number of times.

#68: after "don't stop, decide yourself" the agent still ended turns with an
A/B/C menu. The instruction is now kept with the conversation, re-stated each
turn, and honoured by the plan gate.
"""
import unittest
from types import SimpleNamespace

from strands.types.exceptions import ContextWindowOverflowException, MaxTokensReachedException

from src import pipeline as pl
from src.pipeline import Pipeline
from src.utils import compaction, standing_orders as so


def _tool_msg(n, text):
    return [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{n}", "name": "x", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": f"t{n}", "status": "success",
                                                    "content": [{"text": text}]}}]},
    ]


class StandingAutonomy(unittest.TestCase):
    def test_grant(self):
        for said in ("don't stop, decide yourself", "Stop asking me and just pick one yourself",
                     "you decide", "No more questions please", "Do not stop until it's done",
                     "keep going until done"):
            self.assertTrue(so.granted(said), said)

    def test_plain_requests_do_not_grant(self):
        for said in ("make it warmer", "which template is best for this?", "stop the render",
                     "can you decide between the two sampler options for me and explain"):
            self.assertEqual(so.granted(said), "", said)

    def test_update_keeps_sets_and_lifts(self):
        held = so.update("", "don't stop, decide yourself")
        self.assertTrue(held)
        self.assertEqual(so.update(held, "now make the sky red"), held)       # outlives other messages
        self.assertEqual(so.update(held, "ask me first from now on"), "")     # the user takes it back

    def test_menu_is_found(self):
        ans = ("I found two ways to do this.\n\nA) Use the Flux template\nB) Use the SDXL template\n"
               "C) Build from scratch\n\nWhich one do you want me to go with?")
        self.assertTrue(so.menu(ans))
        self.assertTrue(so.menu("Option A: relight\nOption B: regenerate\n\nWhich option would you like?"))

    def test_report_and_single_question_are_not_menus(self):
        self.assertEqual(so.menu("Done:\n1. saved a.png\n2. saved b.png\n3. saved c.png"), "")
        self.assertEqual(so.menu("Which checkpoint should I use - SD 1.5 or SDXL?"), "")
        self.assertEqual(so.menu(""), "")


class PlanGateHonoursIt(unittest.TestCase):
    def _detect(self, standing, user_text, directive="wait for my go-ahead before running"):
        fake = SimpleNamespace(
            _session=SimpleNamespace(standing_autonomy=standing),
            _canvas_hooks=[{"hook_node_id": 7, "directive": directive}])
        return Pipeline._detect_plan_approval(fake, user_text, "")

    def test_hook_ask_first_holds_without_the_instruction(self):
        self.assertIsNotNone(self._detect("", "make the shots"))

    def test_hook_ask_first_lifted_by_decide_yourself(self):
        self.assertIsNone(self._detect("decide yourself", "make the shots"))

    def test_the_current_message_still_counts(self):
        self.assertIsNotNone(self._detect("decide yourself", "show me the plan first, then wait"))


class MenuIsSentBack(unittest.TestCase):
    def _again(self, standing, answer):
        msgs = [{"role": "assistant", "content": [{"text": answer}]}]
        fake = SimpleNamespace(_orchestrator_agent=SimpleNamespace(messages=msgs),
                               _session=SimpleNamespace(standing_autonomy=standing),
                               _canvas_edit_rejected=None, _verbose=False)
        return Pipeline._unfinished_intent_input(fake, 0)

    MENU = "Two routes:\nA) Flux\nB) SDXL\n\nWhich one do you want?"

    def test_sent_back_when_user_said_decide(self):
        out = self._again("don't stop, decide yourself", self.MENU)
        self.assertIn("decide yourself", out)
        self.assertIn("Which one do you want?", out)

    def test_left_alone_otherwise(self):
        self.assertEqual(self._again("", self.MENU), "")

    def test_redrives_are_bounded(self):
        msgs = [{"role": "assistant", "content": [{"text": self.MENU}]}]
        fake = SimpleNamespace(_orchestrator_agent=SimpleNamespace(messages=msgs),
                               _session=SimpleNamespace(standing_autonomy="decide yourself"),
                               _canvas_edit_rejected=None, _verbose=False)
        self.assertEqual(Pipeline._unfinished_intent_input(fake, 2), "")


class CutOff(unittest.TestCase):
    def _fake(self, messages):
        return SimpleNamespace(_orchestrator_agent=SimpleNamespace(messages=messages),
                               _session=SimpleNamespace(session_id=""), _verbose=False)

    def test_max_tokens_is_redriven_then_gives_up(self):
        fake = self._fake([{"role": "user", "content": [{"text": "go"}]}])
        exc = MaxTokensReachedException("cut")
        first = Pipeline._cut_off_input(fake, exc, 0)
        self.assertIn("output limit", first)
        self.assertIn("NOT finished", first)
        self.assertEqual(Pipeline._cut_off_input(fake, exc, pl._MAX_CUT_OFF_REDRIVES), "")

    def test_overflow_trims_old_tool_output_and_redrives(self):
        msgs = [{"role": "user", "content": [{"text": "go"}]}]
        for n in range(8):
            msgs += _tool_msg(n, "x" * 20_000)
        before = compaction.estimate_tokens(msgs)
        fake = self._fake(msgs)
        out = Pipeline._cut_off_input(fake, ContextWindowOverflowException("full"), 0)
        self.assertIn("no longer fit", out)
        self.assertLess(compaction.estimate_tokens(fake._orchestrator_agent.messages), before)

    def test_overflow_with_nothing_to_trim_is_not_redriven(self):
        fake = self._fake([{"role": "user", "content": [{"text": "go"}]}])
        self.assertEqual(Pipeline._cut_off_input(fake, ContextWindowOverflowException("full"), 0), "")


class StreamHoldsCutOffBack(unittest.IsolatedAsyncioTestCase):
    async def test_exception_lands_in_the_list(self):
        class Agent:
            async def stream_async(self, _inp):
                yield {"data": "partial"}
                raise MaxTokensReachedException("cut")

        fake = SimpleNamespace(_orchestrator_agent=Agent())
        held, seen = [], []
        async for ev in Pipeline._stream_until_cut_off(fake, "go", held):
            seen.append(ev)
        self.assertEqual(seen, [{"data": "partial"}])
        self.assertIsInstance(held[0], MaxTokensReachedException)

    async def test_other_errors_still_propagate(self):
        class Agent:
            async def stream_async(self, _inp):
                raise ValueError("real bug")
                yield  # pragma: no cover

        fake = SimpleNamespace(_orchestrator_agent=Agent())
        with self.assertRaises(ValueError):
            async for _ in Pipeline._stream_until_cut_off(fake, "go", []):
                pass


class SummaryTellsTheAgentToContinue(unittest.TestCase):
    def test_summary_block_carries_the_note_and_it_is_not_refolded(self):
        msgs = []
        for i in range(4):
            msgs += [{"role": "user", "content": [{"text": f"task {i}"}]},
                     {"role": "assistant", "content": [{"text": f"done {i}"}]}]
        new, old = compaction.summarise_older(msgs, keep_turns=1, summarise=lambda m: "GOAL x")
        self.assertTrue(old)
        text = new[0]["content"][0]["text"]
        self.assertIn(compaction.CONTINUE_NOTE, text)
        self.assertNotIn(compaction.CONTINUE_NOTE, compaction.render_for_summary(new))


class PartialsExist(unittest.TestCase):
    def test_partials(self):
        for name in ("autonomy", "decide_yourself", "turn_cut_off"):
            self.assertTrue(pl._orch_partial(name), name)
        self.assertIn("autonomy", pl._TURN_GUIDES)


if __name__ == "__main__":
    unittest.main()
