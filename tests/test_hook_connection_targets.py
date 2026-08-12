"""Tests for delivering a hook's produced value into a CONNECTION input.

A hook output is often wired into a mix of inputs — e.g. one IMAGE plus two
STRING prompts on the same node. The strings are widget literals, but the IMAGE
carries a wire: writing a filename there replaces the link and disconnects the
input, which is the "the prompts arrive but the image is never connected" failure.

Self-contained: hand-built API-format prompts, no ComfyUI.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import (as_connection, build_batch, describe_hooks,
                                    inject_produced_value, is_connection_type)


def _node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _canvas():
    """Two LoadImages (the hook's anchors) and a video node the hook feeds.

    Node 43 mirrors the real case: `first_frame` is an IMAGE connection, while
    `model.prompt` / `model.negative_prompt` are converted widget inputs.
    """
    return {
        "12": _node("LoadImage", image="mb_test_00020_.png", upload="image"),
        "15": _node("LoadImage", image="mb_test_00016_.png", upload="image"),
        # spliced state: the hook's first anchor took over its consumers
        "43": _node("KlingVideo", first_frame=["12", 0],
                    **{"model.prompt": "old", "model.negative_prompt": "old neg"}),
        "50": _node("SaveVideo", video=["43", 0]),
    }


class ConnectionTypeTests(unittest.TestCase):
    def test_primitive_vs_connection(self):
        for t in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO", "string"):
            self.assertFalse(is_connection_type(t), t)
        for t in ("IMAGE", "LATENT", "MODEL", "MASK", "AUDIO", "CONDITIONING"):
            self.assertTrue(is_connection_type(t), t)


class AsConnectionTests(unittest.TestCase):
    def test_node_id_is_wired_directly(self):
        g = _canvas()
        self.assertEqual(as_connection(g, "15", ["12", 0]), ["15", 0])

    def test_explicit_link_passes_through(self):
        g = _canvas()
        self.assertEqual(as_connection(g, ["15", 0], None), ["15", 0])

    def test_filename_reuses_the_node_already_loading_it(self):
        g = _canvas()
        before = set(g)
        self.assertEqual(as_connection(g, "mb_test_00016_.png", ["12", 0]), ["15", 0])
        self.assertEqual(set(g), before, "should not have added a node")

    def test_filename_matches_on_basename(self):
        g = _canvas()
        self.assertEqual(as_connection(g, r"C:\in\mb_test_00016_.png", ["12", 0]), ["15", 0])

    def test_unknown_file_clones_the_current_source(self):
        g = _canvas()
        link = as_connection(g, "brand_new.png", ["12", 0])
        self.assertIsNotNone(link)
        nid = link[0]
        self.assertNotIn(nid, ("12", "15"))
        self.assertEqual(g[nid]["class_type"], "LoadImage")     # kept the user's loader
        self.assertEqual(g[nid]["inputs"]["image"], "brand_new.png")
        self.assertEqual(g["12"]["inputs"]["image"], "mb_test_00020_.png",
                         "cloning must leave the original alone")

    def test_unresolvable_value_returns_none(self):
        g = _canvas()
        # No current wire to clone from and not an image file → cannot be a link.
        self.assertIsNone(as_connection(g, "just some prose", None))


class BuildBatchConnectionTests(unittest.TestCase):
    def _resolutions(self, image_values):
        return [
            {"target_node_id": "43", "param": "first_frame", "mode": "value_list",
             "values": image_values, "zip_group": "pair"},
            {"target_node_id": "43", "param": "model.prompt", "mode": "value_list",
             "values": ["prompt A", "prompt B"], "zip_group": "pair"},
            {"target_node_id": "43", "param": "model.negative_prompt",
             "mode": "value_list", "values": ["neg A", "neg B"], "zip_group": "pair"},
        ]

    def test_image_is_wired_not_written_as_a_literal(self):
        """The regression: filenames used to land in `first_frame` as strings."""
        prompts, notes = build_batch(_canvas(), self._resolutions(
            ["mb_test_00020_.png", "mb_test_00016_.png"]))
        self.assertEqual(len(prompts), 2)
        for p in prompts:
            wire = p["43"]["inputs"]["first_frame"]
            self.assertIsInstance(wire, list, f"first_frame must stay a wire, got {wire!r}")
            self.assertEqual(len(wire), 2)
            self.assertIn(wire[0], p, "wired to a node that exists")
        self.assertEqual(notes, [])

    def test_paired_image_and_prompts_stay_aligned(self):
        prompts, _ = build_batch(_canvas(), self._resolutions(["12", "15"]))
        got = [(p["43"]["inputs"]["first_frame"],
                p["43"]["inputs"]["model.prompt"],
                p["43"]["inputs"]["model.negative_prompt"]) for p in prompts]
        self.assertEqual(got, [(["12", 0], "prompt A", "neg A"),
                               (["15", 0], "prompt B", "neg B")])

    def test_filenames_resolve_to_the_matching_anchors(self):
        prompts, _ = build_batch(_canvas(), self._resolutions(
            ["mb_test_00020_.png", "mb_test_00016_.png"]))
        self.assertEqual([p["43"]["inputs"]["first_frame"] for p in prompts],
                         [["12", 0], ["15", 0]])

    def test_string_inputs_are_still_written_literally(self):
        prompts, _ = build_batch(_canvas(), self._resolutions(["12", "15"]))
        self.assertEqual(prompts[0]["43"]["inputs"]["model.prompt"], "prompt A")

    def test_unresolvable_connection_keeps_the_wire_and_notes_it(self):
        g = _canvas()
        g["43"]["inputs"]["first_frame"] = ["12", 0]
        prompts, notes = build_batch(g, [
            {"target_node_id": "43", "param": "first_frame", "mode": "value_list",
             "values": ["not a node or a file"]},
        ])
        self.assertEqual(prompts[0]["43"]["inputs"]["first_frame"], ["12", 0])
        self.assertTrue(any("connection input" in n for n in notes), notes)

    def test_per_variant_clone_does_not_leak_between_prompts(self):
        prompts, _ = build_batch(_canvas(), [
            {"target_node_id": "43", "param": "first_frame", "mode": "value_list",
             "values": ["new_one.png", "new_two.png"]},
        ])
        files = [p[p["43"]["inputs"]["first_frame"][0]]["inputs"]["image"]
                 for p in prompts]
        self.assertEqual(files, ["new_one.png", "new_two.png"])


class InjectProducedValueTests(unittest.TestCase):
    def test_keep_live_injection_wires_a_connection_target(self):
        g = _canvas()
        hook = {"hook_node_id": "99", "targets": [
            {"node_id": "43", "to_input": "first_frame", "to_input_type": "IMAGE",
             "type": "KlingVideo"}]}
        written = inject_produced_value(g, hook, "15")
        self.assertEqual(written, ["43"])
        self.assertEqual(g["43"]["inputs"]["first_frame"], ["15", 0])

    def test_keep_live_injection_still_writes_strings(self):
        g = _canvas()
        hook = {"hook_node_id": "99", "targets": [
            {"node_id": "43", "to_input": "model.prompt", "to_input_type": "STRING",
             "type": "KlingVideo"}]}
        self.assertEqual(inject_produced_value(g, hook, "hello"), ["43"])
        self.assertEqual(g["43"]["inputs"]["model.prompt"], "hello")


class DescribeHooksTests(unittest.TestCase):
    def test_connection_targets_are_flagged_with_the_anchors(self):
        hooks = [{
            "hook_node_id": "99", "directive": "one image per run, matching prompts",
            "purpose": "inline_parameter",
            "anchors": [{"node_id": "12", "type": "LoadImage", "widgets": {}},
                        {"node_id": "15", "type": "LoadImage", "widgets": {}}],
            "targets": [
                {"node_id": "43", "to_input": "first_frame", "to_input_type": "IMAGE",
                 "type": "KlingVideo"},
                {"node_id": "43", "to_input": "model.prompt", "to_input_type": "STRING",
                 "type": "KlingVideo"},
            ],
        }]
        block = describe_hooks(hooks, _canvas())
        self.assertIn("CONNECTION", block)
        self.assertIn("connect one of 12, 15", block)
        # The STRING target must NOT be flagged as a connection.
        self.assertEqual(block.count("CONNECTION"), 1, block)


if __name__ == "__main__":
    unittest.main()
