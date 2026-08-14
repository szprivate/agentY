"""A hook is spliced out; what replaces it has to be able to feed the input.

From the run of 2026-08-14 20:00. Three queued variants, and every re-queue came
back::

    400 Bad Request … node 348 (ByteDanceSeedreamNodeV2): Return type mismatch
    between linked nodes (model.images.image_1,
    received_type(STRING) mismatch input_type(IMAGE))

The submitted graph had `model.images.image_1` wired to node 75, a
`PrimitiveStringMultiline` — the hook's FIRST anchor, which carried the prompts.
Splicing prefers the anchor whose type matches the target and fell back to
"whatever was wired first" when none did.

None did because of the ref notes. The user had wired LoadImage → agentY ref note
→ the hook's anchor, and the frontend reported each anchor's type from the link
that *arrives* at the hook — which starts at the note and carries its wildcard.
So five image anchors were announced as untyped, the IMAGE target matched none of
them, and the fallback handed it a string.

Both halves are fixed: the type now comes from the node the note wraps, and a
connection target with nothing suitable on the hook is left unwired rather than
wired to something that cannot work.

    python -m unittest discover -s tests
"""

import unittest

from src.utils.canvas_hooks import splice_hook_nodes


def _graph():
    """The user's graph, reduced: hook 5 feeds both the prompt and the image input."""
    return {
        "6": {"class_type": "LoadImage", "inputs": {"image": "hero.png"}},
        "7": {"class_type": "LoadImage", "inputs": {"image": "mentor.jpg"}},
        "75": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "the shots"}},
        "298": {"class_type": "AgentYRefNote", "inputs": {"input": ["6", 0], "role": "HERO"}},
        "5": {"class_type": "AgentYHook",
              "inputs": {"directive": "one pass per prompt",
                         "anchors.anchor0": ["75", 0],      # the prompts — wired first
                         "anchors.anchor1": ["298", 0],     # an image, through a note
                         "anchors.anchor2": ["7", 0]}},
        "348": {"class_type": "ByteDanceSeedreamNodeV2",
                "inputs": {"prompt": ["5", 0], "model.images.image_1": ["5", 0]}},
        "11": {"class_type": "bEpicSendToViewer", "inputs": {"input": ["348", 0]}},
    }


def _hook(anchor_types):
    """The hook payload, with each anchor's reported output type."""
    return {
        "hook_node_id": "5", "purpose": "inline_parameter", "directive": "one pass per prompt",
        "anchors": [
            {"node_id": "75", "from_output_type": "STRING", "to_input": "anchors.anchor0"},
            {"node_id": "6", "from_output_type": anchor_types, "to_input": "anchors.anchor1"},
            {"node_id": "7", "from_output_type": "IMAGE", "to_input": "anchors.anchor2"},
        ],
        "targets": [
            {"node_id": "348", "to_input": "prompt", "to_input_type": "STRING"},
            {"node_id": "348", "to_input": "model.images.image_1", "to_input_type": "IMAGE"},
        ],
    }


class SpliceTypeTest(unittest.TestCase):
    def test_the_string_anchor_is_never_wired_into_the_image_input(self):
        """The failure as it happened: the note-wrapped anchor reported no type."""
        clean, _ = splice_hook_nodes(_graph(), [_hook("COMFY_MATCHTYPE_V3")])
        wired = clean["348"]["inputs"].get("model.images.image_1")
        self.assertNotEqual(wired, ["75", 0],
                            "a prompt string cannot feed an IMAGE input")
        self.assertIn(wired, (["298", 0], ["7", 0]),
                      "either the note (a passthrough) or the image behind it")

    def test_the_note_is_recognised_as_the_image_it_wraps(self):
        """The link names the note; the hook payload names node 6 behind it."""
        clean, _ = splice_hook_nodes(_graph(), [_hook("IMAGE")])
        self.assertEqual(clean["348"]["inputs"]["model.images.image_1"], ["298", 0],
                         "the user's own wiring is preserved — the note passes it through")

    def test_with_nothing_suitable_the_input_is_left_unwired(self):
        """Better a missing input than one wired to something that can't feed it."""
        hook = _hook("STRING")
        hook["anchors"] = [a for a in hook["anchors"] if a["node_id"] == "75"]
        graph = _graph()
        graph["5"]["inputs"] = {"directive": "x", "anchors.anchor0": ["75", 0]}
        clean, _ = splice_hook_nodes(graph, [hook])
        self.assertNotIn("model.images.image_1", clean["348"]["inputs"])

    def test_a_widget_target_still_drops_the_link_for_the_agent_to_fill(self):
        clean, _ = splice_hook_nodes(_graph(), [_hook("IMAGE")])
        self.assertNotIn("prompt", clean["348"]["inputs"],
                         "a STRING target is a value to write, not a wire")

    def test_an_older_frontend_that_reports_no_types_behaves_as_before(self):
        hook = _hook("")
        for a in hook["anchors"]:
            a["from_output_type"] = ""
        clean, _ = splice_hook_nodes(_graph(), [hook])
        self.assertEqual(clean["348"]["inputs"]["model.images.image_1"], ["75", 0],
                         "with nothing known, the first anchor is still the answer")

    def test_a_hook_to_hook_wildcard_target_takes_the_first_anchor(self):
        graph = _graph()
        graph["30"] = {"class_type": "AgentYHook",
                       "inputs": {"directive": "next", "anchors.anchor1": ["5", 0]}}
        hook = _hook("IMAGE")
        hook["targets"].append({"node_id": "30", "to_input": "anchors.anchor1",
                                "to_input_type": "*"})
        clean, _ = splice_hook_nodes(graph, [hook, {"hook_node_id": "30", "anchors": [],
                                                    "targets": []}])
        self.assertNotIn("5", clean, "both hooks are spliced out")

    def test_an_empty_value_unwires_a_connection_input(self):
        """"…otherwise leave the image inputs empty" has to be expressible."""
        from src.utils.canvas_hooks import build_batch
        graph = {"6": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
                 "348": {"class_type": "ByteDanceSeedreamNodeV2",
                         "inputs": {"prompt": "x", "model.images.image_1": ["6", 0]}}}
        prompts, notes = build_batch(graph, [
            {"target_node_id": "348", "param": "model.images.image_1",
             "mode": "value_list", "values": ["6", ""]}])
        self.assertEqual(prompts[0]["348"]["inputs"]["model.images.image_1"], ["6", 0])
        self.assertNotIn("model.images.image_1", prompts[1]["348"]["inputs"],
                         "the second pass asked for no reference")
        self.assertFalse([n for n in notes if "could not wire" in n])

    def test_the_types_that_promise_nothing_fit_anything(self):
        from src.utils.canvas_hooks import _type_fits
        for wildcard in ("", "*", "COMFY_MATCHTYPE_V3", "COMFY_MULTITYPE_V3"):
            with self.subTest(wildcard=wildcard):
                self.assertTrue(_type_fits(wildcard, "IMAGE"))
        self.assertTrue(_type_fits("IMAGE", "IMAGE"))
        self.assertTrue(_type_fits("IMAGE,MASK", "MASK"), "a union offers both")
        self.assertFalse(_type_fits("STRING", "IMAGE"))
        self.assertFalse(_type_fits("LATENT", "IMAGE"))


if __name__ == "__main__":
    unittest.main()
