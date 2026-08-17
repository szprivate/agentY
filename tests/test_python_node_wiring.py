"""Wiring the agentY python node: the slot is `inputs.in0`, never `in0`.

From a real failure: a baked graph came back with

    TypeError: AgentYPython.execute() got an unexpected keyword argument 'in0'

The node's autogrow container is called `inputs` with template prefix `in`, so
ComfyUI expands the slots to `inputs.in0`, `inputs.in1`, … — the same shape a
hook's anchors already use (`anchors.anchor0`). The bake step was naming them
`in0`. That matches no declared input, so ComfyUI handed it to execute() as a
loose keyword and the run died before the snippet ran at all.

Confirmed against the running ComfyUI, which reports the container as:

    ["COMFY_AUTOGROW_V3", {"template": {"input": {"required": {"in": ["*", {}]}},
                           "prefix": "in", "min": 0, "max": 20}}]

    python -m unittest discover -s tests
"""

import re
import unittest

from src.utils.subgraph_bake import build_baked_workflow


def _stage(n_inputs=2):
    """One baked stage with a computed output fed by *n_inputs* inner outputs."""
    return {
        "name": "Measure",
        "hook_node_id": "30",
        "graph": {
            "nodes": [
                {"id": 1, "type": "LoadImage", "pos": [0, 0], "size": [200, 100],
                 "inputs": [], "outputs": [{"name": "IMAGE", "type": "IMAGE",
                                            "links": None}],
                 "widgets_values": ["a.png"]},
                {"id": 2, "type": "SaveImage", "pos": [300, 0], "size": [200, 100],
                 "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                 "outputs": [], "widgets_values": ["out"]},
            ],
            "links": [],
        },
        "computed_outputs": [{
            "name": "length",
            "type": "FLOAT",
            "code": "outputs = [len(in0)]",
            "inputs": [{"node_id": 1, "output_slot": i} for i in range(n_inputs)],
        }],
    }


def _python_node(baked):
    for node in baked.get("definitions", {}).get("subgraphs", [{}])[0].get("nodes", []):
        if node.get("type") == "AgentYPython":
            return node
    for node in baked.get("nodes", []):
        if node.get("type") == "AgentYPython":
            return node
    return None


class SlotNameTest(unittest.TestCase):

    def setUp(self):
        self.node = _python_node(build_baked_workflow([_stage()]))
        self.assertIsNotNone(self.node, "the bake step should inject an AgentYPython")

    def test_the_slots_are_addressed_through_their_container(self):
        names = [i.get("name") for i in self.node.get("inputs") or []]
        self.assertEqual(names, ["inputs.in0", "inputs.in1"])

    def test_a_bare_in0_is_exactly_what_broke_it(self):
        """The name ComfyUI cannot match, and so passes on as a loose keyword."""
        names = [i.get("name") for i in self.node.get("inputs") or []]
        self.assertNotIn("in0", names)
        for name in names:
            self.assertTrue(re.fullmatch(r"inputs\.in\d+", str(name)), name)

    def test_it_matches_the_shape_the_hook_anchors_already_use(self):
        """`anchors.anchor0` and `inputs.in0` are the same rule, not two rules."""
        names = [i.get("name") for i in self.node.get("inputs") or []]
        self.assertTrue(all("." in str(n) for n in names))

    def test_one_input_is_named_the_same_way(self):
        node = _python_node(build_baked_workflow([_stage(n_inputs=1)]))
        self.assertEqual([i.get("name") for i in node.get("inputs") or []],
                         ["inputs.in0"])

    def test_a_computed_output_with_no_inputs_wires_nothing(self):
        node = _python_node(build_baked_workflow([_stage(n_inputs=0)]))
        self.assertEqual(node.get("inputs"), [])

    def test_the_snippet_is_still_carried_verbatim(self):
        self.assertEqual(self.node.get("widgets_values"), ["outputs = [len(in0)]"])

    def test_every_slot_is_actually_linked(self):
        """A named slot with no link is a slot the snippet reads as missing."""
        links = {i.get("link") for i in self.node.get("inputs") or []}
        self.assertNotIn(None, links)
        self.assertEqual(len(links), 2, "two inputs, two distinct links")


if __name__ == "__main__":
    unittest.main()
