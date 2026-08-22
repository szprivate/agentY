"""Filling an input the graph does not show, because nothing is wired to it.

The run this comes from: a hook batch on a Seedream canvas, and the user asking
for three of the images to be redone with a reference photo wired into the node's
image slot. The agent looked, saw ``wired_inputs: {"prompt": …}``, concluded the
node had no image input it could reach, and — twice — queued the runs anyway with
the reference's FILENAME pasted into the prompt text. Both batches reported
``ok``. No image was ever handed to the model.

Two separate holes made that the best available answer:

* an unwired input is **absent** from the API graph, so nothing the agent could
  read said the ten image slots existed at all;
* ``build_batch`` decided "does this input take a wire?" from the HOOKS, which
  only know the inputs they themselves feed — so a resolution aimed at an unwired
  slot fell through to writing a literal string into an ``IMAGE`` input. Accepted
  by ComfyUI, ignored by the node, reported by nobody.

The machinery to wire one was already there (``as_connection`` builds a loader and
returns a link). Only the permission to use it was missing.

    python -m unittest discover -s tests
"""

import asyncio
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from pipeline_stub import pipeline_stub, tools
from src.utils.canvas_hooks import (build_batch, canonical_param,
                                    declared_inputs, open_inputs)
from src.utils.canvas_view import node_detail

REF = "ref.png"          # a bare name: core LoadImage takes it, no ComfyUI needed


# Trimmed to shape from the live /object_info for ByteDanceSeedreamNodeV2. Two
# options, because how many image slots exist DEPENDS on the model selected —
# 5.0 lite really does declare fourteen where 5.0 pro declares ten.
def _slots(n):
    return ["COMFY_AUTOGROW_V3",
            {"template": {"input": {"required": {"image": ["IMAGE", {}]}},
                          "names": [f"image_{i}" for i in range(1, n + 1)],
                          "min": 0}}]


SEEDREAM = {
    "input": {
        "required": {
            "prompt": ["STRING", {}],
            "seed": ["INT", {}],
            "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "seedream 5.0 pro", "inputs": {"required": {
                    "width": ["INT", {}], "images": _slots(10)}}},
                {"key": "seedream 5.0 lite", "inputs": {"required": {
                    "width": ["INT", {}], "images": _slots(14)}}},
            ]}],
        },
        "optional": {"mask": ["MASK", {}]},
    },
}

# The old-style schema shape: a COMBO is an inline LIST of the choices, not a
# type name. Unhashable, so anything testing it against a set of names has to
# check what it is first.
LEGACY = {
    "input": {"required": {
        "ckpt_name": [["sd15.safetensors", "sdxl.safetensors"], {}],
        "image": ["IMAGE", {}],
        "steps": ["INT", {}],
    }},
}

# A node with the same short slot name under TWO groups — the case where the
# short form genuinely cannot be resolved.
AMBIGUOUS = {
    "input": {"required": {
        "a": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "only", "inputs": {"required": {"first": _slots(2)}}}]}],
        "b": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "only", "inputs": {"required": {"second": _slots(2)}}}]}],
    }},
}


def schemas(**over):
    """Patch the schema lookup; an unknown class answers {} (ComfyUI can't say)."""
    table = {"ByteDanceSeedreamNodeV2": SEEDREAM, "Ambiguous": AMBIGUOUS,
             "LegacyNode": LEGACY}
    table.update(over)
    return mock.patch("src.utils.preflight._schema",
                      side_effect=lambda c: table.get(c, {}))


def blind():
    """ComfyUI unreachable: every class answers {}."""
    return mock.patch("src.utils.preflight._schema", side_effect=lambda c: {})


def canvas(model="seedream 5.0 pro", **extra):
    node = {"class_type": "ByteDanceSeedreamNodeV2",
            "inputs": {"prompt": "a room", "seed": 0, "model": model}}
    node["inputs"].update(extra)
    return {"6": node,
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}}}


def res(param, values, **kw):
    out = {"target_node_id": "6", "param": param, "mode": "value_list",
           "values": list(values)}
    out.update(kw)
    return out


class SchemaReadingTest(unittest.TestCase):
    """What the node declares, addressed the way the API prompt addresses it."""

    def test_the_numbered_slots_are_found_and_dotted(self):
        with schemas():
            declared = declared_inputs("ByteDanceSeedreamNodeV2")
        self.assertEqual(declared["model.images.image_1"], "IMAGE")
        self.assertEqual(declared["prompt"], "STRING")
        self.assertEqual(declared["model.width"], "INT")

    def test_a_pure_group_is_not_itself_an_input(self):
        """The prompt carries `model.images.image_1`, never `model.images`."""
        with schemas():
            declared = declared_inputs("ByteDanceSeedreamNodeV2")
        self.assertNotIn("model.images", declared)

    def test_a_dynamic_combo_is_an_input_and_holds_the_option_key(self):
        """`model = "seedream 5.0 pro"` is a real key in every captured graph.

        It contributes its option's inputs under its own name AND carries a value
        of its own, so leaving it out would make this an incomplete description of
        what the node accepts — and nothing could use it to check a built graph.
        """
        with schemas():
            declared = declared_inputs("ByteDanceSeedreamNodeV2")
        self.assertEqual(declared["model"], "COMBO")
        with schemas():
            self.assertNotIn("model", open_inputs(canvas(), "6"),
                             "a combo is not something to wire a loader into")

    def test_which_slots_exist_follows_the_model_selected(self):
        """Merging the options would offer image_12 on a node that has ten."""
        with schemas():
            pro = open_inputs(canvas("seedream 5.0 pro"), "6")
            lite = open_inputs(canvas("seedream 5.0 lite"), "6")
        self.assertIn("model.images.image_10", pro)
        self.assertNotIn("model.images.image_11", pro)
        self.assertIn("model.images.image_14", lite)

    def test_an_unrecognised_selection_falls_back_to_the_first_option(self):
        with schemas():
            slots = open_inputs(canvas("something else entirely"), "6")
        self.assertIn("model.images.image_10", slots)
        self.assertNotIn("model.images.image_11", slots)

    def test_a_wired_slot_is_not_open(self):
        graph = canvas(**{"model.images.image_1": ["9", 0]})
        with schemas():
            slots = open_inputs(graph, "6")
        self.assertNotIn("model.images.image_1", slots)
        self.assertIn("model.images.image_2", slots)

    def test_only_connection_inputs_are_offered(self):
        """A prompt is not something to wire a loader into."""
        with schemas():
            slots = open_inputs(canvas(), "6")
        self.assertNotIn("prompt", slots)
        self.assertNotIn("seed", slots)
        self.assertEqual(slots.get("mask"), "MASK")

    def test_an_old_style_combo_is_a_list_and_does_not_blow_up(self):
        """A pre-V3 schema spells a COMBO as the list of choices itself.

        It is unhashable, so testing it against a set of type NAMES raises
        `TypeError: unhashable type: 'list'` — which is how this got in.
        """
        graph = {"6": {"class_type": "LegacyNode", "inputs": {"steps": 20}}}
        with schemas():
            declared = declared_inputs("LegacyNode")
            slots = open_inputs(graph, "6")
        self.assertEqual(declared["ckpt_name"], "COMBO")
        self.assertEqual(declared["image"], "IMAGE")
        self.assertEqual(slots, {"image": "IMAGE"},
                         "a checkpoint list is not something to wire into")

    def test_a_comfyui_that_cannot_be_asked_says_nothing(self):
        with blind():
            self.assertEqual(declared_inputs("ByteDanceSeedreamNodeV2"), {})
            self.assertEqual(open_inputs(canvas(), "6"), {})


class CanonicalParamTest(unittest.TestCase):
    """`image_1` is what everyone says; `model.images.image_1` is what it is."""

    def test_the_short_name_resolves_to_the_address(self):
        with schemas():
            self.assertEqual(canonical_param(canvas(), "6", "image_1"),
                             "model.images.image_1")

    def test_the_address_is_left_alone(self):
        with schemas():
            self.assertEqual(
                canonical_param(canvas(), "6", "model.images.image_3"),
                "model.images.image_3")

    def test_a_name_the_graph_already_holds_is_left_alone(self):
        with schemas():
            self.assertEqual(canonical_param(canvas(), "6", "prompt"), "prompt")

    def test_a_key_this_graph_really_uses_is_never_moved(self):
        """The compatibility guarantee, and the reason the graph is asked first.

        If a canvas does serialise the slot bare — an older capture, a node that
        simply names it that way — renaming it would write the value to a SECOND
        key and leave the original wire sitting there. What the graph already
        holds beats what the schema would have called it.
        """
        graph = canvas(**{"image_1": ["9", 0]})
        with schemas():
            self.assertEqual(canonical_param(graph, "6", "image_1"), "image_1")

    def test_a_slot_this_model_does_not_have_is_left_alone(self):
        """Ten slots on pro. `image_12` is an honest failure, not a guess."""
        with schemas():
            self.assertEqual(canonical_param(canvas("seedream 5.0 pro"), "6",
                                             "image_12"), "image_12")
            self.assertEqual(canonical_param(canvas("seedream 5.0 lite"), "6",
                                             "image_12"), "model.images.image_12")

    def test_an_ambiguous_short_name_is_left_alone(self):
        """Two groups declare `image_1`; picking one would be a coin toss."""
        graph = {"6": {"class_type": "Ambiguous", "inputs": {}}}
        with schemas():
            self.assertEqual(canonical_param(graph, "6", "image_1"), "image_1")

    def test_nothing_is_renamed_when_comfyui_cannot_be_asked(self):
        with blind():
            self.assertEqual(canonical_param(canvas(), "6", "image_1"), "image_1")


class WiringAnEmptySlotTest(unittest.TestCase):
    """The fix itself: a resolution may fill an input no hook feeds."""

    def _build(self, resolutions, base=None, **kw):
        base = base if base is not None else canvas()
        notes: list = []
        with schemas():
            prompts, notes = build_batch(copy.deepcopy(base), resolutions,
                                         connection_inputs=set(), **kw)
        return prompts, notes

    def test_a_file_becomes_a_loader_wired_into_the_slot(self):
        prompts, _ = self._build([res("model.images.image_1", [REF])])
        graph, = prompts
        wire = graph["6"]["inputs"]["model.images.image_1"]
        self.assertIsInstance(wire, list, "the reference was not wired in")
        loader = graph[wire[0]]
        self.assertEqual(loader["class_type"], "LoadImage")
        self.assertEqual(loader["inputs"]["image"], REF)

    def test_the_short_slot_name_works_too(self):
        prompts, _ = self._build([res("image_1", [REF])])
        graph, = prompts
        self.assertIn("model.images.image_1", graph["6"]["inputs"])
        self.assertNotIn("image_1", graph["6"]["inputs"],
                         "written to a key the node never reads")

    def test_the_literal_is_never_written_into_an_image_input(self):
        """The regression this whole file is about, stated as one assertion."""
        prompts, _ = self._build([res("image_1", [REF])])
        for value in prompts[0]["6"]["inputs"].values():
            self.assertNotEqual(value, REF,
                                "a filename was written where a wire belongs")

    def test_it_says_the_slot_was_empty_and_the_canvas_is_untouched(self):
        _, notes = self._build([res("image_1", [REF])])
        joined = " ".join(notes)
        self.assertIn("model.images.image_1", joined)
        self.assertIn("had nothing wired to it", joined)
        self.assertIn("canvas the user has open is unchanged", joined)

    def test_an_empty_value_leaves_that_run_without_a_reference(self):
        """"…with the reference for three of them" — the ask, exactly."""
        prompts, _ = self._build([
            res("prompt", ["lab", "scanner", "reels", "room"], zip_group="shot"),
            res("image_1", [REF, REF, REF, ""], zip_group="shot")])
        self.assertEqual(len(prompts), 4)
        wired = [isinstance(g["6"]["inputs"].get("model.images.image_1"), list)
                 for g in prompts]
        self.assertEqual(wired, [True, True, True, False])
        self.assertNotIn("model.images.image_1", prompts[3]["6"]["inputs"],
                         "an empty string was left sitting in an IMAGE input")

    def test_prose_is_refused_rather_than_written(self):
        prompts, notes = self._build(
            [res("image_1", ["the attached reference photo"])])
        self.assertNotIn("model.images.image_1", prompts[0]["6"]["inputs"])
        self.assertIn("prose cannot be wired", " ".join(notes))

    def test_a_node_id_is_wired_straight_through(self):
        base = canvas()
        base["9"] = {"class_type": "LoadImage", "inputs": {"image": "ben.png"}}
        prompts, _ = self._build([res("image_1", ["9"])], base=base)
        self.assertEqual(prompts[0]["6"]["inputs"]["model.images.image_1"],
                         ["9", 0])

    def test_a_file_already_on_the_canvas_reuses_its_loader(self):
        base = canvas()
        base["9"] = {"class_type": "LoadImage", "inputs": {"image": REF}}
        prompts, _ = self._build([res("image_1", [REF])], base=base)
        graph, = prompts
        self.assertEqual(graph["6"]["inputs"]["model.images.image_1"], ["9", 0])
        self.assertEqual(len(graph), 3, "a duplicate loader was added")

    def test_each_variant_gets_its_own_loader(self):
        prompts, _ = self._build([res("image_1", ["a.png", "b.png"])])
        files = [g[g["6"]["inputs"]["model.images.image_1"][0]]["inputs"]["image"]
                 for g in prompts]
        self.assertEqual(files, ["a.png", "b.png"])

    def test_the_users_own_graph_object_is_not_touched(self):
        base = canvas()
        before = copy.deepcopy(base)
        self._build([res("image_1", [REF])], base=base)
        self.assertEqual(base, before)

    def test_a_blind_comfyui_changes_nothing_it_cannot_vouch_for(self):
        """No schema, no licence to wire — and no crash either."""
        with blind():
            prompts, _ = build_batch(canvas(), [res("image_1", [REF])],
                                     connection_inputs=set())
        self.assertEqual(prompts[0]["6"]["inputs"]["image_1"], REF)

    def test_a_hook_fed_input_still_works_as_before(self):
        """The hooks' own answer is not overridden by the schema's."""
        base = canvas()
        base["9"] = {"class_type": "LoadImage", "inputs": {"image": "ben.png"}}
        with schemas():
            prompts, _ = build_batch(base, [res("mask", ["9"])],
                                     connection_inputs={"6.mask"})
        self.assertEqual(prompts[0]["6"]["inputs"]["mask"], ["9", 0])


class NodeDetailTest(unittest.TestCase):
    """What `get_canvas_node` shows — the half of the failure about looking."""

    def test_the_empty_slots_are_reported(self):
        with schemas():
            detail = node_detail(canvas(), "6")
        self.assertIn("model.images.image_1", detail["open_inputs"])
        self.assertIn("NOTHING is wired to them", detail["open_inputs_note"])

    def test_ten_identical_slots_are_one_line(self):
        with schemas():
            detail = node_detail(canvas(), "6")
        self.assertEqual(len(detail["open_inputs"]), 2)     # the images, and mask
        self.assertIn("image_10", detail["open_inputs"]["model.images.image_1"])

    def test_a_fully_wired_node_says_nothing_extra(self):
        graph = {"6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}}}
        with schemas():
            detail = node_detail(graph, "6")
        self.assertNotIn("open_inputs", detail)

    def test_the_old_fields_are_all_still_there(self):
        with schemas():
            detail = node_detail(canvas(**{"model.images.image_1": ["9", 0]}), "6")
        self.assertEqual(detail["values"]["prompt"], "a room")
        self.assertEqual(detail["wired_inputs"]["model.images.image_1"],
                         "from #9 output 0")

    def test_a_blind_comfyui_reports_no_open_inputs(self):
        with blind():
            detail = node_detail(canvas(), "6")
        self.assertNotIn("open_inputs", detail)


class ThroughTheToolTest(unittest.TestCase):
    """End to end: the loader has to survive the trim and reach the file."""

    def _apply(self, resolutions, base=None):
        pipe = pipeline_stub(_canvas_base_prompt=base or canvas(),
                             _canvas_hooks=[], _dry_run=True)
        with schemas(), \
             mock.patch("agenty_core.tools.comfyui.open_workflow_in_canvas"), \
             mock.patch("src.executor._autoload_workflows_into_canvas",
                        return_value=False):
            return json.loads(asyncio.run(
                tools(pipe)["apply_canvas_hooks"](resolutions)))

    def _graphs(self, out):
        return [json.loads(Path(v["workflow"]).read_text(encoding="utf-8"))
                for v in out["variants"]]

    def test_the_reference_reaches_the_workflow_that_runs(self):
        out = self._apply([res("image_1", [REF])])
        self.assertEqual(out["count"], 1)
        graph, = self._graphs(out)
        wire = graph["6"]["inputs"]["model.images.image_1"]
        self.assertIsInstance(wire, list, "the reference was not wired in")
        self.assertIn(wire[0], graph, "the loader was trimmed out of the graph")
        self.assertEqual(graph[wire[0]]["inputs"]["image"], REF)

    def test_three_with_a_reference_and_one_without(self):
        out = self._apply([
            res("prompt", ["lab", "scanner", "reels", "room"], zip_group="shot"),
            res("image_1", [REF, REF, REF, ""], zip_group="shot")])
        self.assertEqual(out["count"], 4)
        wired = [isinstance(g["6"]["inputs"].get("model.images.image_1"), list)
                 for g in self._graphs(out)]
        self.assertEqual(wired, [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
