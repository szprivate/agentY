"""A hook wired into ANOTHER hook's anchor is a chain handoff, not a target to fill.

The reported failure, from the log of 2026-08-13 18:18: hook 44's output was wired
into the anchors of hooks 27 and 5. The block advertised those as ordinary targets —

    PRODUCER hook 44 (context: node 43 (LoadImage) …) feeds node 27 (AgentYHook)'s
    `anchors.anchor1` input, * [CONNECTION: supply a node id — connect one of 43,
    not a value]; node 5 (AgentYHook)'s `anchors.anchor4` input, * [CONNECTION: …]

— and the RUN PLAN told the agent hook 44 had to be RUN this turn. So it called
apply_canvas_hooks against those ids and got back

    {"error": "no batch was produced",
     "notes": ["node 27 is not in the canvas graph — skipped",
               "node 5 is not in the canvas graph — skipped"]}

because hook nodes are spliced out of the graph that runs. Reading that as "the
canvas route doesn't work here", the agent went off to run_info + prepare_workflow
and started BUILDING a workflow for a graph the user already had on the canvas.

The input side of the block always knew about hook→hook wires ("the value you
produce for hook N"); only the output side didn't.

    python -m unittest tests.test_hook_chain_targets
"""

import unittest

from src.utils.canvas_hooks import (build_batch, describe_hooks, gating_hook_ids,
                                    plan_lines)


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """The runnable graph: hook nodes are already spliced out, as at run time."""
    return {
        "43": _node("LoadImage", image="style_ref.jpg", upload="image"),
        "6": _node("LoadImage", image="hero.jpg", upload="image"),
        "242": _node("OpenAIGPTImageNodeV2", prompt="old", seed=1),
        "250": _node("SaveImage", images=["242", 0]),
    }


def _styleguide_hook():
    """Hook 44 — its output only reaches other hooks' anchors."""
    return {
        "hook_node_id": "44", "purpose": "inline_parameter",
        "directive": "analyse the image in anchor_0 and create a STYLEGUIDE from it",
        "anchors": [{"node_id": "43", "type": "LoadImage", "widgets": {}}],
        "targets": [
            {"node_id": "27", "to_input": "anchors.anchor1", "to_input_type": "*",
             "type": "AgentYHook"},
            {"node_id": "5", "to_input": "anchors.anchor4", "to_input_type": "*",
             "type": "AgentYHook"},
        ],
    }


def _gpt_hook():
    """Hook 5 — the real one, feeding the GPT Image 2 node on the canvas."""
    return {
        "hook_node_id": "5", "purpose": "inline_parameter",
        "directive": ("Take the character, place and wardrobe prompts and run one GPT "
                      "Image 2 pass for each. Queue the prompt in three consecutive runs. "
                      "Only continue when all of them succeeded."),
        "anchors": [{"node_id": "6", "type": "LoadImage", "widgets": {}}],
        "targets": [
            {"node_id": "242", "to_input": "model.images.image_1",
             "to_input_type": "IMAGE", "type": "OpenAIGPTImageNodeV2"},
            {"node_id": "242", "to_input": "prompt", "to_input_type": "STRING",
             "type": "OpenAIGPTImageNodeV2"},
        ],
    }


def _hook_27():
    return {"hook_node_id": "27", "purpose": "text", "directive": "write a caption",
            "anchors": [], "targets": []}


class ChainOnlyProducerTests(unittest.TestCase):
    def setUp(self):
        self.hooks = [_styleguide_hook(), _hook_27(), _gpt_hook()]
        self.block = describe_hooks(self.hooks, _canvas())

    def test_a_hook_anchor_is_not_offered_as_a_fillable_target(self):
        # The exact string the agent copied into apply_canvas_hooks.
        self.assertNotIn("node 27 (AgentYHook)", self.block)
        self.assertNotIn("node 5 (AgentYHook)", self.block)
        self.assertNotIn("anchors.anchor1", self.block)
        self.assertNotIn("anchors.anchor4", self.block)

    def test_it_says_who_reads_the_value_instead(self):
        self.assertIn("CHAIN ONLY", self.block)
        self.assertIn("hook 5, hook 27", self.block)
        self.assertIn("read the value you produce here as context", self.block)

    def test_it_points_at_place_canvas_text_and_forbids_the_batch_call(self):
        line = next(l for l in self.block.splitlines() if "PRODUCER hook 44" in l)
        self.assertIn('place_canvas_text(hook_node_id="44"', line)
        self.assertIn("Do NOT call apply_canvas_hooks for this hook", line)

    def test_a_real_target_is_still_described_exactly(self):
        line = next(l for l in self.block.splitlines() if "PRODUCER hook 5" in l)
        self.assertIn("node 242 (OpenAIGPTImageNodeV2)", line)
        self.assertIn("`prompt` input, STRING", line)
        self.assertIn("CONNECTION", line)

    def test_the_run_plan_does_not_demand_a_run_of_a_chain_only_hook(self):
        # Hook 5's directive is conditional, so 44 is upstream of a condition --
        # but there is nothing to execute for 44, and telling the agent to run it
        # is what produced "no batch was produced".
        self.assertNotIn("44", gating_hook_ids(self.hooks))
        plan = "\n".join(plan_lines(self.hooks))
        self.assertNotIn("Hook(s) 44 must be RUN", plan)

    def test_the_producer_rules_rule_out_building(self):
        self.assertIn("do NOT call prepare_workflow", self.block)
        self.assertIn("ALREADY on the canvas", self.block)


class ChainTargetsAreUnrunnableTests(unittest.TestCase):
    """Why the block must not offer them: the tool genuinely cannot serve one."""

    def test_a_sweep_aimed_at_a_hook_node_produces_nothing(self):
        prompts, notes = build_batch(_canvas(), [
            {"target_node_id": "27", "param": "anchors.anchor1",
             "mode": "value_list", "values": ["43"]},
            {"target_node_id": "5", "param": "anchors.anchor4",
             "mode": "value_list", "values": ["43"]},
        ])
        self.assertEqual(prompts, [], "hook nodes are spliced out before the run")
        self.assertTrue(any("not in the canvas graph" in n for n in notes), notes)

    def test_a_sweep_aimed_at_the_real_node_still_works(self):
        prompts, notes = build_batch(_canvas(), [
            {"target_node_id": "242", "param": "prompt", "mode": "value_list",
             "values": ["characters", "places", "wardrobe"]},
        ])
        self.assertEqual([p["242"]["inputs"]["prompt"] for p in prompts],
                         ["characters", "places", "wardrobe"])
        self.assertEqual(notes, [])


class TextHookChainTests(unittest.TestCase):
    """Hook 4 in the log had the same shape and survived only by luck."""

    def test_a_text_hook_reports_its_consumers_not_a_phantom_input(self):
        hooks = [
            {"hook_node_id": "4", "purpose": "text", "directive": "extract the characters",
             "anchors": [{"node_id": "75", "type": "PrimitiveStringMultiline",
                          "widgets": {"value": "a script"}}],
             "targets": [{"node_id": "5", "to_input": "anchors.anchor0",
                          "to_input_type": "*", "type": "AgentYHook"}]},
            _gpt_hook(),
        ]
        block = describe_hooks(hooks, _canvas())
        line = next(l for l in block.splitlines() if "TEXT hook 4" in l)
        self.assertIn("hook 5 read the value you produce here as context", line)
        self.assertNotIn("anchors.anchor0", line)
        self.assertIn("write & place", line)


if __name__ == "__main__":
    unittest.main()
