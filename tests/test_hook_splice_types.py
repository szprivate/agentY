"""Tests for type-aware hook splicing.

A hook's output is commonly wired into a mix of inputs on the same node — one
IMAGE plus two prompt STRINGs in the reported case. Splicing used to rewire every
one of them to the hook's first anchor, which put a LoadImage on the prompt boxes:
inert on a normal run, and indistinguishable from a real connection afterwards.

Self-contained: hand-built API-format prompts, no ComfyUI.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import build_batch, splice_hook_nodes


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """The logged graph: hook 38 reads LoadImage 36/37 and feeds three inputs on 43."""
    return {
        "36": _node("LoadImage", image="mb_test_00020_.png"),
        "37": _node("LoadImage", image="mb_test_00016_.png"),
        "38": _node("AgentYHook", anchor0=["36", 0], anchor1=["37", 0],
                    directive="one image per run with matching prompts"),
        "43": _node("Wan2ImageToVideoApi", first_frame=["38", 0],
                    **{"model.prompt": ["38", 0], "model.negative_prompt": ["38", 0]}),
        "50": _node("SaveVideo", video=["43", 0]),
    }


def _hooks():
    return [{
        "hook_node_id": "38",
        "directive": "one image per run with matching prompts",
        "purpose": "inline_parameter",
        "anchors": [
            {"node_id": "36", "type": "LoadImage", "from_output_type": "IMAGE"},
            {"node_id": "37", "type": "LoadImage", "from_output_type": "IMAGE"},
        ],
        "targets": [
            {"node_id": "43", "to_input": "first_frame", "to_input_type": "IMAGE",
             "type": "Wan2ImageToVideoApi"},
            {"node_id": "43", "to_input": "model.prompt", "to_input_type": "STRING",
             "type": "Wan2ImageToVideoApi"},
            {"node_id": "43", "to_input": "model.negative_prompt",
             "to_input_type": "STRING", "type": "Wan2ImageToVideoApi"},
        ],
    }]


class SpliceByTypeTests(unittest.TestCase):
    def test_image_target_keeps_the_anchor_passthrough(self):
        clean, removed = splice_hook_nodes(_canvas(), _hooks())
        self.assertEqual(removed, ["38"])
        self.assertEqual(clean["43"]["inputs"]["first_frame"], ["36", 0])

    def test_string_targets_are_not_wired_to_an_image_anchor(self):
        """The regression: LoadImage 36 used to land on both prompt inputs."""
        clean, _ = splice_hook_nodes(_canvas(), _hooks())
        inputs = clean["43"]["inputs"]
        self.assertNotIn("model.prompt", inputs)
        self.assertNotIn("model.negative_prompt", inputs)

    def test_produced_prompts_can_then_be_written(self):
        """End to end: the values the agent authored must reach the graph."""
        clean, _ = splice_hook_nodes(_canvas(), _hooks())
        prompts, notes = build_batch(clean, [
            {"target_node_id": "43", "param": "first_frame", "mode": "value_list",
             "values": ["36", "37"], "zip_group": "pair"},
            {"target_node_id": "43", "param": "model.prompt", "mode": "value_list",
             "values": ["prompt A", "prompt B"], "zip_group": "pair"},
            {"target_node_id": "43", "param": "model.negative_prompt",
             "mode": "value_list", "values": ["neg A", "neg B"], "zip_group": "pair"},
        ])
        self.assertEqual(notes, [], "no input should be reported unresolvable now")
        got = [(p["43"]["inputs"]["first_frame"],
                p["43"]["inputs"]["model.prompt"],
                p["43"]["inputs"]["model.negative_prompt"]) for p in prompts]
        self.assertEqual(got, [(["36", 0], "prompt A", "neg A"),
                               (["37", 0], "prompt B", "neg B")])

    def test_anchor_is_matched_by_wire_type_when_several_are_wired(self):
        """A mixed-anchor hook passes the STRING wire through, not the IMAGE one."""
        g = {
            "20": _node("LoadImage", image="a.png"),
            "21": _node("PrimitiveString", value="hello"),
            "38": _node("AgentYHook", anchor0=["20", 0], anchor1=["21", 0],
                        directive="x"),
            "43": _node("Sampler", image=["38", 0], text=["38", 0]),
        }
        hooks = [{
            "hook_node_id": "38",
            "anchors": [{"node_id": "20", "from_output_type": "IMAGE"},
                        {"node_id": "21", "from_output_type": "CONDITIONING"}],
            "targets": [
                {"node_id": "43", "to_input": "image", "to_input_type": "IMAGE"},
                {"node_id": "43", "to_input": "text", "to_input_type": "CONDITIONING"},
            ],
        }]
        clean, _ = splice_hook_nodes(g, hooks)
        self.assertEqual(clean["43"]["inputs"]["image"], ["20", 0])
        self.assertEqual(clean["43"]["inputs"]["text"], ["21", 0])

    def test_without_hook_metadata_behaviour_is_unchanged(self):
        """Older callers pass no hooks — every consumer is rewired, as before."""
        clean, _ = splice_hook_nodes(_canvas())
        inputs = clean["43"]["inputs"]
        self.assertEqual(inputs["first_frame"], ["36", 0])
        self.assertEqual(inputs["model.prompt"], ["36", 0])

    def test_dangling_hook_still_drops_its_consumers(self):
        g = _canvas()
        g["38"]["inputs"] = {"directive": "no anchors"}
        clean, _ = splice_hook_nodes(g, _hooks())
        self.assertNotIn("first_frame", clean["43"]["inputs"])
        self.assertNotIn("38", clean)

    def test_unrelated_links_are_untouched(self):
        clean, _ = splice_hook_nodes(_canvas(), _hooks())
        self.assertEqual(clean["50"]["inputs"]["video"], ["43", 0])


if __name__ == "__main__":
    unittest.main()
