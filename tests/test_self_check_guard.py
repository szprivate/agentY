"""Output checks switched off means the orchestrator does not QA its own renders.

2026-10-29: with Settings > Output checks > qa > enabled OFF, the orchestrator
still ended each turn with analyze_image on everything it had just rendered. The
host's QA honoured the switch; the model's own habit did not.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from agenty_core.utils import turn_scope

from src import pipeline as pl
from src.utils import self_check_guard as g

OUT = [r"\\BEPIC-STORAGE01\work\out\v001\bear_00001_.png",
       r"\\BEPIC-STORAGE01\work\out\v002\bear_00001_.png"]
REF = r"D:\refs\bear_reference.png"


class TheRule(unittest.TestCase):

    def test_off_refuses_a_look_at_only_fresh_renders(self):
        why = g.refusal("analyze_image", {"file_paths": OUT, "question": "rate 1-10"},
                        OUT, ["three more scenes of the bear"], enabled=False)
        self.assertIn("Output checks are switched off", why)
        self.assertTrue(g.refusal("analyze_image", {"file_path": OUT[0]}, OUT, [""], enabled=False))
        self.assertTrue(g.refusal("analyze_video", {"file_path": OUT[1]}, OUT, [""], enabled=False))

    def test_on_lets_it_through(self):
        self.assertEqual(g.refusal("analyze_image", {"file_paths": OUT}, OUT, [""], enabled=True), "")

    def test_the_user_asking_lets_it_through(self):
        for text in ("make three and tell me which one is best", "look at them",
                     "does it match the reference?", "check the hands", "schau sie dir an"):
            self.assertEqual(g.refusal("analyze_image", {"file_paths": OUT}, OUT, [text],
                                       enabled=False), "", text)

    def test_generation_wording_is_not_a_request_to_look(self):
        for text in ("make him look happier", "use the flux checkpoint, 24 frame rate",
                     "the same bear in a Berlin U-Bahn"):
            self.assertTrue(g.refusal("analyze_image", {"file_paths": OUT}, OUT, [text],
                                      enabled=False), text)

    def test_inputs_and_comparisons_are_not_self_checks(self):
        self.assertEqual(g.refusal("analyze_image", {"file_path": REF}, OUT, [""], enabled=False), "")
        self.assertEqual(g.refusal("analyze_image", {"file_paths": [REF, OUT[0]]}, OUT, [""],
                                   enabled=False), "")
        self.assertEqual(g.refusal("analyze_image", {"image_url": "https://x/y.png"}, OUT, [""],
                                   enabled=False), "")
        self.assertEqual(g.refusal("view_image", {"file_path": OUT[0]}, OUT, [""], enabled=False), "")

    def test_a_turn_that_rendered_nothing_is_not_guarded(self):
        self.assertEqual(g.refusal("analyze_image", {"file_path": OUT[0]}, [], [""], enabled=False), "")


class TheHook(unittest.TestCase):

    def _event(self, args, messages=()):
        return SimpleNamespace(tool_use={"name": "analyze_image", "input": args},
                               agent=SimpleNamespace(messages=list(messages)),
                               cancel_tool=False)

    def _in_turn(self, user_text, outputs, event):
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        try:
            g.begin_turn(user_text, lambda: list(outputs))
            with mock.patch.object(g, "checks_on", return_value=False):
                g.SelfCheckGuardHookProvider()._on_before(event)
        finally:
            turn_scope.leave(token)
        return event.cancel_tool

    def test_cancels_the_self_check(self):
        self.assertTrue(self._in_turn("same bear, three scenes", OUT, self._event({"file_paths": OUT})))

    def test_an_interjection_asking_to_look_lets_it_through(self):
        msgs = [{"role": "user", "content": [{"text": "same bear, three scenes"}]},
                {"role": "assistant", "content": [{"toolUse": {"name": "run_variants"}}]},
                {"role": "user", "content": [{"toolResult": {"content": [
                    {"text": "ok"},
                    {"text": "[USER INTERJECTION - ...]\n\nlook at them and pick the best"}]}}]}]
        self.assertFalse(self._in_turn("same bear, three scenes", OUT,
                                       self._event({"file_paths": OUT}, msgs)))

    def test_outside_a_turn_nothing_is_refused(self):
        token = turn_scope.enter(turn_scope.Scope("other", "other"))
        try:
            ev = self._event({"file_paths": OUT})
            with mock.patch.object(g, "checks_on", return_value=False):
                g.SelfCheckGuardHookProvider()._on_before(ev)
            self.assertFalse(ev.cancel_tool)
        finally:
            turn_scope.leave(token)

    def test_the_orchestrator_carries_it(self):
        from src import agent
        self.assertIn("SelfCheckGuardHookProvider", inspect.getsource(agent.create_orchestrator_agent))


class ThePrompt(unittest.TestCase):

    def test_the_guide_goes_in_only_while_checks_are_off(self):
        self.assertIn("output_checks_off", pl._TURN_GUIDES)
        self.assertIn("analyze_image", pl._orch_partial("output_checks_off"))
        session = SimpleNamespace(current_output_paths=[])
        token = turn_scope.enter(turn_scope.Scope("req", "thread"))
        try:
            with mock.patch.object(g, "checks_on", return_value=False):
                self.assertTrue(pl._self_checks_off("hi", session))
            with mock.patch.object(g, "checks_on", return_value=True):
                self.assertFalse(pl._self_checks_off("hi", session))
            session.current_output_paths.append(OUT[0])
            turn = turn_scope.current().get(g.SCOPE_SLOT)
            self.assertEqual(turn["outputs"](), [OUT[0]])          # read live, not copied
        finally:
            turn_scope.leave(token)

    def test_no_tool_text_tells_it_to_qa_its_outputs(self):
        src = inspect.getsource(pl)
        self.assertNotIn("then QA the outputs", src)
        prompt = (pl._ORCH_PARTIALS_DIR.parent / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertNotIn("checking\n  several outputs", prompt.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
