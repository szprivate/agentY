"""Tests for the deterministic hook RUN PLAN (canvas_hooks.plan_lines).

A hook directive can gate the run on how an earlier step turned out ("wait for the
references; if ANY failed, STOP"). Queued work has no results during the turn, so
whatever such a condition reads has to be RUN, not queued. The plan works that out
from the wiring + the directives before the agent starts, because getting it wrong
is silent: queue a batch, reach the conditional hook, stop — and cancel the very
work the condition was about.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import (describe_hooks, gating_hook_ids, is_conditional,
                                    plan_lines)


def _hook(hid, directive, anchors=(), purpose="inline_parameter"):
    return {"hook_node_id": str(hid), "purpose": purpose, "directive": directive,
            "anchors": [{"node_id": str(a)} for a in anchors]}


# The user's real graph, reduced: 44 → 27 → 4 → 5 → 30, where 30 gates on 5.
def _film_hooks():
    return [
        _hook(44, "analyse the image in anchor_0 and describe the STYLE and COLOR.",
              anchors=[43]),
        _hook(4, "Take the connected script, extract all the main characters.",
              anchors=[3], purpose="text"),
        _hook(27, "Break the story in anchor_0 down into single shots.", anchors=[74, 44]),
        _hook(5, "For every place and character in anchor_0, create a prompt for a "
                 "reference frame.", anchors=[6, 7, 8, 4]),
        _hook(30, "Wait for all the references to be generated. If ANY reference "
                  "generation failed - STOP, and ask the user for advice.",
              anchors=[27, 5]),
    ]


class ConditionalDetectionTest(unittest.TestCase):
    def test_the_directive_that_started_this(self):
        self.assertTrue(is_conditional(_hook(
            30, "Wait for all the references to be generated. If ANY reference "
                "generation failed - STOP, and ask the user for advice.")))

    def test_other_shapes_of_the_same_intent(self):
        for directive in [
            "Only continue once every shot exists.",
            "If any of them failed, abort and tell me.",
            "Do not proceed if the upscale errored.",
            "Wait until all three references are complete, then build the sheet.",
            "Halt if the script names no characters.",
        ]:
            with self.subTest(directive=directive):
                self.assertTrue(is_conditional(_hook(9, directive)), directive)

    def test_ordinary_directives_are_not_conditional(self):
        for directive in [
            "Write a caption for the image in anchor_0.",
            "Create one starting image for each of the prompts.",
            "Break the story down into single shots, max 2 seconds per shot.",
            "Describe the style, colour and contrast of this frame.",
            # 'stop' inside an unrelated word must not trip it
            "Render a non-stop dolly move across the courtyard.",
        ]:
            with self.subTest(directive=directive):
                self.assertFalse(is_conditional(_hook(9, directive)), directive)


class GatingTest(unittest.TestCase):
    def test_everything_upstream_of_a_conditional_hook_must_be_run(self):
        gating = gating_hook_ids(_film_hooks())
        # 30 reads 27 and 5; 5 reads 4; 27 reads 44. Hook 4 is a TEXT hook — it
        # writes a string, there is nothing to run early, so it isn't listed.
        self.assertEqual(gating, {"27", "5", "44"})
        self.assertNotIn("30", gating, "the conditional hook itself is not gating")

    def test_no_conditional_hook_means_nothing_has_to_run_early(self):
        hooks = [h for h in _film_hooks() if h["hook_node_id"] != "30"]
        self.assertEqual(gating_hook_ids(hooks), set())

    def test_a_conditional_hook_with_no_upstream_gates_nothing(self):
        self.assertEqual(gating_hook_ids([_hook(1, "Stop if the folder is empty.")]), set())

    def test_a_cycle_does_not_hang(self):
        a = _hook(1, "produce a value", anchors=[2])
        b = _hook(2, "Stop if any failed.", anchors=[1])
        self.assertEqual(gating_hook_ids([a, b]), {"1"})


class ProducerSideWiringTest(unittest.TestCase):
    """The shape the real graph had: hook 30's anchors report 'no input wired'.

    The 44→27 and 27/5→30 links exist only as *targets* on the producing hooks
    ("feeds node 30 (AgentYHook)'s anchors.anchor0"). Reading the consumer side
    alone found no dependencies, so the plan came out empty on exactly the graph
    that needed it.
    """

    @staticmethod
    def _graph():
        def feeds(hid, directive, targets, anchors=()):
            h = _hook(hid, directive, anchors=anchors)
            h["targets"] = [{"node_id": str(t), "to_input": "anchors.anchor0"} for t in targets]
            return h
        return [
            feeds(44, "analyse the image and describe the STYLE.", targets=[27], anchors=[43]),
            feeds(27, "Break the story down into single shots.", targets=[30], anchors=[74]),
            feeds(5, "Create a reference-frame prompt for every character.",
                  targets=[23, 30], anchors=[6, 7, 8]),
            feeds(30, "Wait for all the references to be generated. If ANY reference "
                      "generation failed - STOP, and ask the user for advice.", targets=[42]),
        ]

    def test_dependencies_are_found_from_the_producer_side(self):
        self.assertEqual(gating_hook_ids(self._graph()), {"27", "5", "44"})

    def test_the_plan_names_the_hook_that_must_run(self):
        text = "\n".join(plan_lines(self._graph()))
        self.assertIn("run_now=true", text)
        self.assertIn("5", text)
        self.assertIn("Hook 30 is CONDITIONAL", text)
        self.assertIn("reads hook", text)


class PlanTest(unittest.TestCase):
    def test_the_plan_names_the_hooks_that_must_run_and_the_conditional_one(self):
        text = "\n".join(plan_lines(_film_hooks()))
        self.assertIn("RUN PLAN", text)
        self.assertIn("run_now=true", text)
        self.assertIn("must be RUN THIS TURN, not queued", text)
        self.assertIn("Hook 30 is CONDITIONAL", text)
        self.assertIn("stop_hook_run", text)
        # It has to say why queueing can't work, or the agent repeats the mistake.
        self.assertIn("end of the turn", text)

    def test_no_plan_when_nothing_is_conditional(self):
        hooks = [h for h in _film_hooks() if h["hook_node_id"] != "30"]
        self.assertEqual(plan_lines(hooks), [])
        self.assertEqual(plan_lines([]), [])

    def test_the_plan_reaches_the_hooks_block_the_agent_reads(self):
        block = describe_hooks(_film_hooks(), {})
        self.assertIn("RUN PLAN", block)
        self.assertIn("run_now=true", block)

    def test_an_ordinary_graph_gets_no_plan_noise(self):
        block = describe_hooks([_hook(1, "Caption the image in anchor_0.", anchors=[2])], {})
        self.assertNotIn("RUN PLAN", block)


if __name__ == "__main__":
    unittest.main()
