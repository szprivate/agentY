"""The guidance for a kind of turn is in the prompt, once - not in every message.

A message stays in the history. Guidance that travelled in it was sent again
with every later call, and once more for every turn that was kept: on a canvas
with hooks that was 45,000 characters of the same text per message.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from pipeline_stub import pipeline_stub
from src import pipeline as pl

BASE = "You are the orchestrator."
SKILLS = "\n\n<available_skills>one</available_skills>"


def _stub(prompt=BASE):
    return pipeline_stub(_orchestrator_agent=SimpleNamespace(system_prompt=prompt))


class WhereItGoes(unittest.TestCase):

    def test_into_the_prompt_in_a_fixed_order(self):
        p = _stub()
        self.assertTrue(p._set_turn_guides(["input_images", "canvas_hooks"]))
        got = p._orchestrator_agent.system_prompt
        self.assertTrue(got.startswith(BASE + pl._TURN_GUIDES_HEAD))
        hooks, images = pl._orch_partial("canvas_hooks"), pl._orch_partial("input_images")
        self.assertTrue(hooks and images)
        self.assertLess(got.index(hooks), got.index(images))       # the table's order, not the caller's

    def test_the_same_turn_again_is_the_same_prompt(self):
        p = _stub()
        p._set_turn_guides(["canvas_hooks", "review_halt"])
        first = p._orchestrator_agent.system_prompt
        p._set_turn_guides(["review_halt", "canvas_hooks"])
        self.assertEqual(p._orchestrator_agent.system_prompt, first)   # a cache holds it

    def test_a_turn_of_another_kind_takes_it_out_again(self):
        p = _stub()
        p._set_turn_guides(["canvas_hooks"])
        p._set_turn_guides([])
        self.assertEqual(p._orchestrator_agent.system_prompt, BASE)

    def test_what_was_added_behind_it_stays(self):
        """The skills plugin appends to the prompt after this has run."""
        p = _stub()
        p._set_turn_guides(["canvas_nodes"])
        p._orchestrator_agent.system_prompt += SKILLS
        p._set_turn_guides(["input_images"])
        got = p._orchestrator_agent.system_prompt
        self.assertIn(SKILLS, got)
        self.assertNotIn(pl._orch_partial("canvas_nodes"), got)
        self.assertIn(pl._orch_partial("input_images"), got)
        self.assertEqual(got.count(pl._TURN_GUIDES_HEAD), 1)

    def test_without_a_prompt_to_put_it_in_it_says_so(self):
        self.assertFalse(pipeline_stub()._set_turn_guides(["canvas_hooks"]))
        self.assertFalse(_stub(prompt=None)._set_turn_guides(["canvas_hooks"]))

    def test_it_can_be_switched_back_to_the_message(self):
        p = _stub()
        with mock.patch.dict(os.environ, {"AGENTY_GUIDES_INLINE": "1"}):
            self.assertFalse(p._set_turn_guides(["canvas_hooks"]))
        self.assertEqual(p._orchestrator_agent.system_prompt, BASE)


class OllamaModel:        # named as the real one is: the class name is what is asked
    pass


class ALocalOrchestrator(unittest.TestCase):
    """qwen3.8:27b did worse with the guidance away from what it is about."""

    def _local(self, prompt=BASE):
        return pipeline_stub(_orchestrator_agent=SimpleNamespace(system_prompt=prompt, model=OllamaModel()))

    def test_keeps_it_in_the_message(self):
        p = self._local()
        self.assertFalse(p._set_turn_guides(["canvas_nodes"]))
        self.assertEqual(p._orchestrator_agent.system_prompt, BASE)

    def test_and_what_a_hosted_turn_left_in_the_prompt_comes_out(self):
        p = _stub()
        p._set_turn_guides(["canvas_nodes"])
        p._orchestrator_agent.model = OllamaModel()
        self.assertFalse(p._set_turn_guides(["canvas_nodes"]))
        self.assertEqual(p._orchestrator_agent.system_prompt, BASE)

    def test_earlier_turns_lose_theirs(self):
        guide = pl._orch_partial("canvas_hooks")
        review = pl._orch_partial("review_halt")
        msgs = [{"role": "user", "content": [{"text": guide + "\n\n[CANVAS HOOKS] 4 -> 7\n" + review + "\n\nrun it"}]},
                {"role": "assistant", "content": [{"text": guide}]},
                {"role": "user", "content": [{"toolResult": {"content": [{"text": "x"}]}}]}]
        self.assertEqual(pl.Pipeline._drop_old_guides(msgs), 2)
        self.assertEqual(msgs[0]["content"][0]["text"], "[CANVAS HOOKS] 4 -> 7\nrun it")
        self.assertEqual(msgs[1]["content"][0]["text"], guide)          # only what the pipeline put in
        self.assertEqual(pl.Pipeline._drop_old_guides(msgs), 0)


class TheMessage(unittest.TestCase):

    def _input(self, p):
        p._extract_hard_constraints = lambda text: []
        p._describe_canvas_selection = lambda: "[CANVAS SELECTION] node 3"
        p._get_memory_context = lambda text: ""
        p._get_project_memory_context = lambda: ""
        p._detect_plan_approval = lambda text, ctx: None
        p._prepend_gallery = lambda text: text
        p._annotate_attachments = lambda user_input, text: text
        with mock.patch("src.executor._autoload_workflows_into_canvas", return_value=True):
            return p._build_orchestrator_input("make it blue", "make it blue")

    def test_it_carries_the_facts_and_not_the_guidance(self):
        p = _stub()
        p._canvas_base_prompt = None
        got = self._input(p)
        guide = pl._orch_partial("canvas_nodes")
        self.assertIn("[CANVAS SELECTION] node 3", got)
        self.assertNotIn(guide, got)
        self.assertIn(guide, p._orchestrator_agent.system_prompt)

    def test_where_the_prompt_cannot_take_it_the_message_does(self):
        p = pipeline_stub()
        p._canvas_base_prompt = None
        got = self._input(p)
        self.assertIn(pl._orch_partial("canvas_nodes"), got)
        self.assertIn("[CANVAS SELECTION] node 3", got)


class AClaudeOrchestrator(unittest.TestCase):

    def test_the_prompt_of_the_call_is_the_agents_not_the_one_the_model_was_built_with(self):
        from src.agent import AnthropicModel
        model = AnthropicModel(model_id="claude-haiku-4-5", max_tokens=1024, client_args={"api_key": "x"},
                               params={"system": [{"type": "text", "text": "as built",
                                                   "cache_control": {"type": "ephemeral"}}]})
        req = model.format_request([{"role": "user", "content": [{"text": "hi"}]}], None, "as built, and more")
        self.assertEqual(req["system"], [{"type": "text", "text": "as built, and more",
                                          "cache_control": {"type": "ephemeral"}}])
        req = model.format_request([{"role": "user", "content": [{"text": "hi"}]}], None, None)
        self.assertEqual(req["system"][0]["text"], "as built")


if __name__ == "__main__":
    unittest.main()
